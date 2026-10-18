"""FraudGuard Phase 1.5: synthetic time-respecting flow and hold recommendations.

Research/demo only. No payment rails, live accounts, or real transaction holds.
The module distinguishes oracle evaluation (uses synthetic labels for penalties)
from deployable-style optimization (uses only observable risk scores for costs).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set
import random
import time

@dataclass(frozen=True)
class Edge:
    edge_id: str
    source: str
    target: str
    timestamp: int
    amount: float
    is_fraud_related: bool  # hidden synthetic evaluation label; never used by estimated optimizer
    risk: float             # observable score, intentionally overlapping across classes
    hold_seconds: int = 30

@dataclass
class NetworkScenario:
    edges: List[Edge]
    seed_accounts: Dict[str, float]
    cashout_accounts: Set[str]
    mule_accounts: Set[str]
    account_is_legitimate: Dict[str, bool]
    metadata: dict

@dataclass
class SimulationResult:
    fraud_flow_reaching_cashout: float
    suspected_flow_created: float
    held_edge_ids: List[str]
    legitimate_edges_held: int
    legitimate_value_held: float
    fraud_related_edges_held: int
    propagated_edge_flow: Dict[str, float]

@dataclass
class ContainmentResult:
    strategy: str
    held_edge_ids: List[str]
    baseline_cashout_flow: float
    remaining_cashout_flow: float
    fraud_flow_contained: float
    legitimate_edges_held: int
    legitimate_value_held: float
    objective_value: float
    runtime_ms: float
    rounds: int
    candidate_count: int = 0
    estimated_objective_value: Optional[float] = None
    def to_dict(self): return asdict(self)


def generate_scenario(n_accounts=80, n_edges=240, mule_share=0.12,
                      fraud_seed_count=3, forwarding_probability=0.78,
                      legitimate_share=0.93, seed=42) -> NetworkScenario:
    """Generate a hard, synthetic case with mostly legitimate edges and overlapping risk.

    Assumptions: seeded suspected funds; transfer events are timestamped; random
    edges are mostly legitimate; a small set of explicitly labelled edges create
    fraud-related routes. Scores overlap: legitimate scores ~U(.02,.82), fraud
    scores ~U(.18,.98), so a single threshold cannot perfectly separate classes.
    Some legitimate edges touch high-degree merchant/salary-style accounts.
    """
    if n_accounts < 8: raise ValueError('n_accounts must be at least 8')
    if n_edges < 1: raise ValueError('n_edges must be positive')
    if not 0 <= mule_share < .8: raise ValueError('mule_share must be in [0,.8)')
    if not 0 <= legitimate_share <= .99: raise ValueError('legitimate_share must be in [0,.99]')
    if not 0 <= forwarding_probability <= 1: raise ValueError('forwarding_probability must be in [0,1]')
    rng = random.Random(seed)
    accounts = [f'A{i:04d}' for i in range(n_accounts)]
    shuffled = accounts[:]; rng.shuffle(shuffled)
    mules = set(shuffled[:max(1, int(n_accounts*mule_share))])
    non_mules = [a for a in accounts if a not in mules]
    cashouts = set(rng.sample(non_mules, max(1, min(len(non_mules)//5, int(n_accounts*.08)))))
    possible_seeds = [a for a in non_mules if a not in cashouts]
    seed_accounts = {a: float(rng.randint(5000, 25000)) for a in rng.sample(possible_seeds, min(fraud_seed_count, len(possible_seeds)))}
    merchants = set(rng.sample(non_mules, max(1, min(len(non_mules), n_accounts//12))))
    edges: List[Edge] = []
    for i in range(n_edges):
        is_fraud = rng.random() > legitimate_share
        if is_fraud:
            source = rng.choice(list(seed_accounts) or non_mules)
            target = rng.choice(list(mules))
            timestamp = rng.randint(1, 45)
        else:
            # Busy legitimate accounts create realistic centrality confounders.
            source = rng.choice(list(merchants)) if rng.random() < .35 else rng.choice(accounts)
            target = rng.choice(accounts)
            timestamp = rng.randint(1, 120)
        if source == target:
            target = accounts[(accounts.index(source)+1) % len(accounts)]
        amount = round(rng.lognormvariate(7.5, .8), 2)
        amount = min(50000.0, max(25.0, amount))
        risk = rng.uniform(.18, .98) if is_fraud else rng.uniform(.02, .82)
        edges.append(Edge(f'E{i:06d}', source, target, timestamp, amount, is_fraud, round(risk, 4), rng.choice([15,30,60])))
    # Ensure every seed has a time-respecting route to a mule then a cash-out.
    # These edges are a small, known fraction and still receive overlapping scores.
    next_id = n_edges
    for source in seed_accounts:
        mule = rng.choice(list(mules)); cashout = rng.choice(list(cashouts))
        # Some edges on a suspected-flow route are ordinary/ambiguous transfers:
        # the label is not equivalent to whether suspected funds traverse the edge.
        # Every planted route edge is fraud-related: labels describe the planted
        # route, not a coin flip independent of the scenario construction.
        edges.append(Edge(f'E{next_id:06d}', source, mule, 1, min(seed_accounts[source], 12000.0), True, round(rng.uniform(.30,.90),4), 30)); next_id += 1
        edges.append(Edge(f'E{next_id:06d}', mule, cashout, 2, min(seed_accounts[source]*.7, 9000.0), True, round(rng.uniform(.30,.90),4), 30)); next_id += 1
    # Legitimate decoys: salary, merchant settlement, rent and bill payments
    # touch reachable mule accounts. They are ordinary customer transfers,
    # not part of the planted stolen-fund route, and create measurable collateral.
    decoy_types = ['salary', 'bill', 'rent', 'merchant_settlement']
    decoys = max(4, min(n_edges // 12, n_accounts // 2))
    for j in range(decoys):
        mule = rng.choice(list(mules))
        if j % 2 == 0:
            source, target = rng.choice(non_mules), mule
        else:
            source, target = mule, rng.choice(non_mules)
        if source == target:
            target = accounts[(accounts.index(source)+1) % len(accounts)]
        amount = round(min(25000.0, max(25.0, rng.lognormvariate(7.2, .65))), 2)
        edges.append(Edge(f'E{next_id:06d}', source, target, rng.randint(1, 120),
                          amount, False, round(rng.uniform(.10, .88),4), rng.choice([15,30,60])))
        next_id += 1
    edges.sort(key=lambda e:(e.timestamp,e.edge_id))
    return NetworkScenario(edges, seed_accounts, cashouts, mules,
        {a:a not in mules for a in accounts},
        {'seed':seed,'n_accounts':n_accounts,'n_edges':len(edges),'legitimate_share_requested':legitimate_share,
         'mule_share_requested':mule_share,'forwarding_probability':forwarding_probability,
         'split_fraction':.65,'forwarding_delay_rule':'incoming flow is available only to strictly later timestamp events',
         'cashout_rule':'flow entering a cash-out account is terminal','merchant_accounts':sorted(merchants),
         'risk_overlap_rule':'legitimate U(.02,.82), fraud U(.18,.98); explicit path edges use overlapping U(.38,.97)'})


def simulate_propagation(scenario, held_edge_ids=None, forwarding_probability=None,
                         split_fraction=.65, seed=7, frozen_accounts=None):
    """Propagate flow in timestamp order; held edges are removed for this window.

    `frozen_accounts` is an explicit optional freeze policy: when an account is
    frozen, outgoing transfers from it are suppressed. This is a simulation only.
    Per-edge random draws are deterministic and paired across candidate sets.
    """
    if not 0 <= split_fraction <= 1: raise ValueError('split_fraction must be in [0,1]')
    p = scenario.metadata.get('forwarding_probability',.78) if forwarding_probability is None else forwarding_probability
    if not 0 <= p <= 1: raise ValueError('forwarding_probability must be in [0,1]')
    held=set(held_edge_ids or []); frozen=set(frozen_accounts or [])
    balance=dict(scenario.seed_accounts); last_received={a:-1 for a in scenario.seed_accounts}
    edge_flow={}; cashout=0.0; created=float(sum(scenario.seed_accounts.values()))
    by_id={e.edge_id:e for e in scenario.edges}
    for edge in sorted(scenario.edges,key=lambda e:(e.timestamp,e.edge_id)):
        if edge.edge_id in held or edge.source in frozen: continue
        available=balance.get(edge.source,0.0)
        if available<=0 or last_received.get(edge.source,-1)>=edge.timestamp: continue
        if random.Random(f'{seed}:{edge.edge_id}').random()>p: continue
        flow=min(float(edge.amount),available*split_fraction)
        if flow<=0: continue
        balance[edge.source]=max(0.0,available-flow); edge_flow[edge.edge_id]=flow
        if edge.target in scenario.cashout_accounts: cashout+=flow
        else:
            balance[edge.target]=balance.get(edge.target,0.0)+flow
            last_received[edge.target]=edge.timestamp
    legitimate=[by_id[e] for e in held if e in by_id and not by_id[e].is_fraud_related]
    fraud=[by_id[e] for e in held if e in by_id and by_id[e].is_fraud_related]
    return SimulationResult(round(cashout,4),round(created,4),sorted(held),len(legitimate),round(sum(e.amount for e in legitimate),4),len(fraud),{k:round(v,4) for k,v in edge_flow.items()})


def _true_objective(sim, lc, lv):
    return sim.fraud_flow_reaching_cashout + lc*sim.legitimate_edges_held + lv*sim.legitimate_value_held

def _estimated_objective(flow, held_edges, lc, lv):
    # No hidden is_fraud_related labels: expected legitimate disruption is based on 1-risk.
    return flow + sum(lc*(1-e.risk) + lv*e.amount*(1-e.risk) for e in held_edges)

def _estimated_flow(scenario, held, p, split, seed=None):
    """Deterministic expected-value proxy; does not share evaluator random draws.

    Each edge receives expected flow = min(amount, available * split) * p * risk.
    This is a deliberately simple deployable-style estimate, not a calibrated
    probabilistic money-flow model. The seed argument is accepted for API
    compatibility but does not affect this deterministic calculation.
    """
    held=set(held)
    balance=dict(scenario.seed_accounts)
    last_received={a:-1 for a in scenario.seed_accounts}
    cashout=0.0
    for e in sorted(scenario.edges,key=lambda x:(x.timestamp,x.edge_id)):
        if e.edge_id in held:
            continue
        avail=balance.get(e.source,0.0)
        if avail<=0 or last_received.get(e.source,-1)>=e.timestamp:
            continue
        expected_flow=min(e.amount,avail*split)*p*e.risk
        if expected_flow<=0:
            continue
        balance[e.source]=max(0.0,avail-expected_flow)
        if e.target in scenario.cashout_accounts:
            cashout+=expected_flow
        else:
            balance[e.target]=balance.get(e.target,0.0)+expected_flow
            last_received[e.target]=e.timestamp
    return cashout


def greedy_containment(scenario, k=3, lambda_count=250., lambda_value=.03,
                       forwarding_probability=None, split_fraction=.65, seed=7,
                       mode='estimated', candidate_limit=None):
    """Greedy optimizer.

    mode='estimated' uses risk-weighted proxy flow and expected legitimate cost,
    never hidden labels. mode='oracle' uses the simulation's true synthetic labels
    and is an upper-bound diagnostic only. Candidate pruning keeps top-ranked
    edges by risk/amount; candidate_limit=None evaluates all edges.
    """
    if k<0: raise ValueError('k must be non-negative')
    if mode not in ('estimated','oracle'): raise ValueError("mode must be 'estimated' or 'oracle'")
    start=time.perf_counter(); p=scenario.metadata.get('forwarding_probability',.78) if forwarding_probability is None else forwarding_probability
    all_edges=list(scenario.edges)
    if candidate_limit is not None and candidate_limit>0 and candidate_limit<len(all_edges):
        # Include high risk and high-amount candidates, deduplicated; no labels used.
        ranked=sorted(all_edges,key=lambda e:(e.risk, e.amount),reverse=True)
        chosen={e.edge_id:e for e in ranked[:candidate_limit]}
        for e in sorted(all_edges,key=lambda e:e.amount,reverse=True)[:max(1,candidate_limit//4)]: chosen[e.edge_id]=e
        candidates=list(chosen.values())
    else: candidates=all_edges
    held=set(); current_true=simulate_propagation(scenario,[],p,split_fraction,seed)
    if mode=='oracle': current_obj=_true_objective(current_true,lambda_count,lambda_value)
    else: current_obj=_estimated_objective(_estimated_flow(scenario,held,p,split_fraction,seed),[],lambda_count,lambda_value)
    rounds=0; current_est=current_obj
    while rounds<k and candidates:
        best=None; best_obj=current_obj
        for edge in candidates:
            if edge.edge_id in held: continue
            trial=held|{edge.edge_id}
            if mode=='oracle':
                sim=simulate_propagation(scenario,trial,p,split_fraction,seed); val=_true_objective(sim,lambda_count,lambda_value)
            else:
                flow=_estimated_flow(scenario,trial,p,split_fraction,seed)
                val=_estimated_objective(flow,[e for e in candidates if e.edge_id in trial],lambda_count,lambda_value)
            if val<best_obj-1e-8: best,best_obj=edge.edge_id,val
        if best is None: break
        held.add(best); current_obj=best_obj; rounds+=1
    true_final=simulate_propagation(scenario,held,p,split_fraction,seed)
    baseline=simulate_propagation(scenario,[],p,split_fraction,seed)
    est_final=_estimated_objective(_estimated_flow(scenario,held,p,split_fraction,seed),[e for e in all_edges if e.edge_id in held],lambda_count,lambda_value)
    return ContainmentResult('greedy_'+mode+('_pruned' if len(candidates)<len(all_edges) else ''),sorted(held),baseline.fraud_flow_reaching_cashout,true_final.fraud_flow_reaching_cashout,
        round(baseline.fraud_flow_reaching_cashout-true_final.fraud_flow_reaching_cashout,4),true_final.legitimate_edges_held,true_final.legitimate_value_held,
        round(current_obj,4),(time.perf_counter()-start)*1000,rounds,len(candidates),round(est_final,4))

def _result_for_held(scenario, held, name, lc, lv, seed=7):
    sim=simulate_propagation(scenario,held,seed=seed); base=simulate_propagation(scenario,seed=seed)
    return ContainmentResult(name,sorted(held),base.fraud_flow_reaching_cashout,sim.fraud_flow_reaching_cashout,
        round(base.fraud_flow_reaching_cashout-sim.fraud_flow_reaching_cashout,4),sim.legitimate_edges_held,sim.legitimate_value_held,
        round(_true_objective(sim,lc,lv),4),0.,len(held),len(scenario.edges),None)

def evaluate_baselines(scenario,k=3,lambda_count=250.,lambda_value=.03,seed=7,candidate_limit=None):
    rng=random.Random(seed); edges=scenario.edges
    ids_random={e.edge_id for e in rng.sample(edges,min(k,len(edges)))}
    ids_risk={e.edge_id for e in sorted(edges,key=lambda e:(e.risk,e.amount),reverse=True)[:k]}
    degree={}
    for e in edges: degree[e.source]=degree.get(e.source,0)+1; degree[e.target]=degree.get(e.target,0)+1
    ids_degree={e.edge_id for e in sorted(edges,key=lambda e:(degree.get(e.source,0)+degree.get(e.target,0),e.risk),reverse=True)[:k]}
    results=[_result_for_held(scenario,set(),'no_holds',lambda_count,lambda_value,seed),
        _result_for_held(scenario,ids_random,'random',lambda_count,lambda_value,seed),
        _result_for_held(scenario,ids_risk,'risk_ranked',lambda_count,lambda_value,seed),
        _result_for_held(scenario,ids_degree,'degree_centrality',lambda_count,lambda_value,seed),
        greedy_containment(scenario,k,lambda_count,lambda_value,seed=seed,mode='estimated',candidate_limit=candidate_limit),
        greedy_containment(scenario,k,lambda_count,lambda_value,seed=seed,mode='oracle')]
    return results

def exact_optimum(scenario,k=3,lambda_count=250.,lambda_value=.03,max_candidates=18,seed=7):
    edges=scenario.edges
    if len(edges)>max_candidates: raise ValueError(f'exact_optimum limited to {max_candidates} edges; got {len(edges)}')
    start=time.perf_counter(); base=simulate_propagation(scenario,seed=seed); best=set(); bestsim=base; bestobj=_true_objective(base,lambda_count,lambda_value)
    for size in range(1,min(k,len(edges))+1):
        for combo in combinations([e.edge_id for e in edges],size):
            sim=simulate_propagation(scenario,combo,seed=seed); val=_true_objective(sim,lambda_count,lambda_value)
            if val<bestobj-1e-9: best=set(combo); bestsim=sim; bestobj=val
    return ContainmentResult('exact_optimum_oracle',sorted(best),base.fraud_flow_reaching_cashout,bestsim.fraud_flow_reaching_cashout,
        round(base.fraud_flow_reaching_cashout-bestsim.fraud_flow_reaching_cashout,4),bestsim.legitimate_edges_held,bestsim.legitimate_value_held,
        round(bestobj,4),(time.perf_counter()-start)*1000,len(best),len(edges),None)

def lambda_sweep(scenario, lambdas, k=3, lambda_count=250., seed=7, candidate_limit=80,
                lambda_counts=None):
    """Single-scenario compatibility sweep over value penalty (legacy helper)."""
    rows=[]
    for lv in lambdas:
        r=greedy_containment(scenario,k,lambda_count,lv,seed=seed,mode='estimated',candidate_limit=candidate_limit)
        rows.append({'lambda_count':float(lambda_count),'lambda_value':float(lv),
                     'fraud_flow_contained':r.fraud_flow_contained,
                     'remaining_cashout_flow':r.remaining_cashout_flow,
                     'legitimate_edges_held':r.legitimate_edges_held,
                     'legitimate_value_held':r.legitimate_value_held,
                     'runtime_ms':r.runtime_ms,'held_edge_ids':r.held_edge_ids,
                     'estimated_objective_value':r.estimated_objective_value})
    return rows


def multi_seed_lambda_sweep(seeds, lambda_counts, lambda_values, n_accounts=80, n_edges=240,
                            k=3, candidate_limit=80, mule_share=.12, legitimate_share=.93):
    """Compare containment settings across independently generated scenarios."""
    import statistics

    rows = []
    for lc in lambda_counts:
        for lv in lambda_values:
            per = []
            for seed in seeds:
                scenario = generate_scenario(
                    n_accounts=n_accounts, n_edges=n_edges,
                    mule_share=mule_share, legitimate_share=legitimate_share,
                    seed=seed
                )
                result = greedy_containment(
                    scenario, k, lc, lv, seed=seed, mode="estimated",
                    candidate_limit=candidate_limit
                )
                per.append(result)

            def stats(attr):
                values = [float(getattr(item, attr)) for item in per]
                return {
                    "mean": round(statistics.mean(values), 4),
                    "sd": round(statistics.pstdev(values), 4)
                }

            hold_sets = [
                tuple(sorted(item.held_edge_ids)) for item in per
            ]
            hold_counts = [len(item.held_edge_ids) for item in per]

            rows.append({
                "lambda_count": float(lc),
                "lambda_value": float(lv),
                "seed_count": len(per),
                "contained_flow": stats("fraud_flow_contained"),
                "legitimate_edges_held": stats("legitimate_edges_held"),
                "legitimate_value_held": stats("legitimate_value_held"),
                "estimated_objective": stats("estimated_objective_value"),
                "held_edge_count": {
                    "mean": round(statistics.mean(hold_counts), 4),
                    "sd": round(statistics.pstdev(hold_counts), 4)
                },
                "unique_hold_sets": len(set(hold_sets)),
                "runtime_ms": stats("runtime_ms")
            })

    return rows


def pruning_report(seeds, candidate_limits, n_accounts=80, edge_counts=(300,600,1200),
                   k=3, lambda_count=250., lambda_value=.03):
    """Compare full candidate search with pruned search over multiple seeds.

    The report explicitly presents speed/quality trade-offs; it does not imply
    pruning is lossless. Quality is measured by actual simulated containment.
    """
    import statistics
    rows=[]
    for n_edges in edge_counts:
        for seed in seeds:
            scenario=generate_scenario(n_accounts=n_accounts,n_edges=n_edges,seed=seed)
            full=greedy_containment(scenario,k,lambda_count,lambda_value,seed=seed,
                                    mode='estimated',candidate_limit=None)
            for limit in candidate_limits:
                pruned=greedy_containment(scenario,k,lambda_count,lambda_value,seed=seed,
                                          mode='estimated',candidate_limit=limit)
                rows.append({'n_edges':n_edges,'seed':seed,'candidate_limit':limit,
                             'full_contained':full.fraud_flow_contained,
                             'pruned_contained':pruned.fraud_flow_contained,
                             'containment_delta':pruned.fraud_flow_contained-full.fraud_flow_contained,
                             'full_runtime_ms':full.runtime_ms,'pruned_runtime_ms':pruned.runtime_ms,
                             'speedup':round(full.runtime_ms/max(pruned.runtime_ms,1e-9),3)})
    # Per-size/limit aggregates make this directly reportable.
    summary=[]
    for n_edges in edge_counts:
        for limit in candidate_limits:
            group=[r for r in rows if r['n_edges']==n_edges and r['candidate_limit']==limit]
            deltas=[r['containment_delta'] for r in group]
            speeds=[r['speedup'] for r in group]
            summary.append({'n_edges':n_edges,'candidate_limit':limit,'seed_count':len(group),
                            'mean_containment_delta':round(statistics.mean(deltas),4),
                            'sd_containment_delta':round(statistics.pstdev(deltas),4),
                            'mean_speedup':round(statistics.mean(speeds),3),
                            'sd_speedup':round(statistics.pstdev(speeds),3)})
    return {'per_run':rows,'summary':summary}
