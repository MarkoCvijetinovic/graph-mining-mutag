# Graph Mining on MUTAG

This project explores classical graph mining techniques on the MUTAG dataset:
- Ullmann subgraph isomorphism
- Topological graph descriptors
- Classification (Logistic Regression, Naive Bayes)
- Unsupervised clustering (OPTICS)
- Graph and cluster visualization

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python graph_utils.py
python graph_iso.py
python graph_desc.py
python graph_ml.py
