# Graph Mining (MUTAG)

- Loads MUTAG dataset (HuggingFace)
- Converts PyG graphs to NetworkX
- Implements graph isomorphism
- Computes topological descriptors
- Runs classification (LogReg / Naive Bayes) + clustering (OPTICS)
- Visualizes molecules and clusters

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

