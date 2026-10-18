# FraudGuard – Real-Time AI for Critical Decisions (IBM Z Datathon)

Real-time fraud scoring: stream -> FastAPI /score -> ML model -> approve / verify / block + reasons -> live dashboard + audit log.

## Run (on IBM LinuxONE)
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python train.py                          # trains model (synthetic data, or data/paysim.csv if present)
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    python simulator.py 20                   # second terminal: 20 tx/s
    python benchmark.py                      # latency numbers for your IBM Z slide

## View the dashboard from your laptop
    ssh -L 8000:localhost:8000 linux1@<your-linuxone-host>
    open http://localhost:8000

## Real data
Download PaySim, rename to data/paysim.csv, re-run train.py. Columns used:
step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,isFraud
