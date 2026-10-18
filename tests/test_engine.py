import unittest
from network.engine import (Edge, NetworkScenario, generate_scenario, simulate_propagation,
                            greedy_containment, exact_optimum, lambda_sweep)

class Phase16Tests(unittest.TestCase):
    def test_deterministic_generation(self):
        a=generate_scenario(n_accounts=24,n_edges=50,seed=19)
        b=generate_scenario(n_accounts=24,n_edges=50,seed=19)
        self.assertEqual(a.edges,b.edges)
        self.assertEqual(a.seed_accounts,b.seed_accounts)

    def test_generator_is_mostly_legitimate_and_risk_overlaps(self):
        s=generate_scenario(n_accounts=80,n_edges=1000,legitimate_share=.93,seed=2)
        ordinary=s.edges[:1000]
        self.assertGreater(sum(not e.is_fraud_related for e in ordinary)/len(ordinary),.85)
        legit=[e.risk for e in ordinary if not e.is_fraud_related]
        fraud=[e.risk for e in ordinary if e.is_fraud_related]
        self.assertTrue(any(x>.4 for x in legit))
        self.assertTrue(any(x<.7 for x in fraud))

    def test_flow_respects_time_order(self):
        s=NetworkScenario([Edge('late','M','C',10,1000,True,.9),Edge('early','S','M',20,1000,True,.9)],
            {'S':1000},{'C'},{'M'},{'S':True,'M':False,'C':True},{'forwarding_probability':1.0})
        r=simulate_propagation(s,forwarding_probability=1.0,split_fraction=1.0,seed=1)
        self.assertEqual(r.fraud_flow_reaching_cashout,0.0)

    def test_hold_budget_and_estimated_mode(self):
        s=generate_scenario(n_accounts=30,n_edges=50,seed=5)
        r=greedy_containment(s,k=2,seed=9,mode='estimated',candidate_limit=20)
        self.assertLessEqual(len(r.held_edge_ids),2)
        self.assertIn('estimated',r.strategy)
        self.assertIsNotNone(r.estimated_objective_value)

    def test_oracle_mode_is_explicit(self):
        s=generate_scenario(n_accounts=20,n_edges=15,seed=3)
        r=greedy_containment(s,k=1,seed=4,mode='oracle')
        self.assertIn('oracle',r.strategy)

    def test_exact_is_no_worse_than_oracle_greedy(self):
        s=generate_scenario(n_accounts=8,n_edges=4,mule_share=.125,fraud_seed_count=1,seed=2)
        g=greedy_containment(s,k=1,seed=3,mode='oracle')
        ex=exact_optimum(s,k=1,max_candidates=12,seed=3)
        self.assertLessEqual(ex.objective_value,g.objective_value+1e-6)

    def test_lambda_sweep_returns_tradeoff_points(self):
        s=generate_scenario(n_accounts=24,n_edges=30,seed=7)
        rows=lambda_sweep(s,[0,.01,.1],k=1,seed=2,candidate_limit=10)
        self.assertEqual(len(rows),3)
        self.assertTrue(all('legitimate_value_held' in r for r in rows))

    def test_rerouting_is_not_claimed_monotonic(self):
        s=generate_scenario(n_accounts=24,n_edges=80,seed=11)
        baseline=simulate_propagation(s,seed=5)
        self.assertGreaterEqual(baseline.fraud_flow_reaching_cashout,0)

    def test_planted_route_edges_are_labelled_fraud_and_decoys_legitimate(self):
        s=generate_scenario(n_accounts=40,n_edges=120,seed=22)
        route_edges=[e for e in s.edges if e.edge_id in {f'E{i:06d}' for i in range(120,126)}]
        self.assertTrue(route_edges)
        self.assertTrue(all(e.is_fraud_related for e in route_edges))
        decoys=[e for e in s.edges if not e.is_fraud_related and
                (e.source in s.mule_accounts or e.target in s.mule_accounts)]
        self.assertGreater(len(decoys),0)

    def test_estimated_mode_ignores_hidden_labels(self):
        s=generate_scenario(n_accounts=32,n_edges=60,seed=41)
        original=greedy_containment(s,k=2,seed=12,mode='estimated',candidate_limit=None)
        flipped=[Edge(e.edge_id,e.source,e.target,e.timestamp,e.amount,not e.is_fraud_related,e.risk,e.hold_seconds)
                 for e in s.edges]
        s2=NetworkScenario(flipped,s.seed_accounts,s.cashout_accounts,s.mule_accounts,
                           s.account_is_legitimate,s.metadata)
        changed=greedy_containment(s2,k=2,seed=12,mode='estimated',candidate_limit=None)
        self.assertEqual(original.held_edge_ids,changed.held_edge_ids)

    def test_multi_seed_lambda_sweep_and_pruning_report(self):
        from network.engine import multi_seed_lambda_sweep, pruning_report
        sweep=multi_seed_lambda_sweep(seeds=[1,2],lambda_counts=[0,250],
             lambda_values=[0,.1],n_accounts=24,n_edges=25,k=1,candidate_limit=12)
        self.assertEqual(len(sweep),4)
        self.assertTrue(all(x['seed_count']==2 for x in sweep))
        report=pruning_report(seeds=[1,2],candidate_limits=[10],n_accounts=24,
                              edge_counts=(20,),k=1)
        self.assertEqual(len(report['summary']),1)
        self.assertEqual(report['summary'][0]['seed_count'],2)

if __name__=='__main__': unittest.main()
