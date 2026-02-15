# Graph Mining on MUTAG

This project explores classical graph mining techniques on the MUTAG dataset, with both
algorithmic implementations and educational material.

Implemented methods include:
- Ullmann subgraph isomorphism
- Topological graph descriptors
- Classification (Logistic Regression, Naive Bayes, SVM)
- Unsupervised clustering (OPTICS)
- Graph and cluster visualization

In addition to the code, this repository contains **presentation slides (Serbian and English)**
used for a lecture on graph mining and an intuitive introduction to Graph Neural Networks (GNNs).
The slides explain both classical graph-based approaches and modern neural methods, with examples
drawn from the MUTAG dataset.

## Repository structure
- `graph_utils.py` – dataset loading, conversions, and visualization utilities
- `graph_iso.py` – subgraph isomorphism (Ullmann) and matching experiments
- `graph_desc.py` – extraction of topological graph descriptors
- `graph_ml.py` – classification and clustering on graph-level features
- `presentations/` – lecture slides (Serbian and English)

## Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Linux/macOS)
source .venv/bin/activate
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run program
python graph_utils.py
python graph_iso.py
python graph_desc.py
python graph_ml.py
