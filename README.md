# Selective Adversarial Training with Encoding Network for Review-Based Cross-Domain Recommendation

### Project Structure

- Source & Target domain dataset
  - (can be excuted with additional downloads: http://jmcauley.ucsd.edu/data/amazon/)
  - We substitute Baby.json > Musical_Instruments.json
  - Musical_Instruments.json, Pation_Lawn_and_Garden.json
- preprocess.py
  - Pre-process source & target domain data
- model.py
- main.py
  - Main file, load the files below

### Setup

```bash
# PyTorch Conda
pip install -r requirements.txt

# Download pretrained word embeddings, GloVe (below)
# https://nlp.stanford.edu/data/glove.6B.zip
# Unzipped file name should be './glove.6B.100d.txt'

```

### Usage

```bash

python3 ./SER/main.py  # Training / Test with following code
```

### Experiments

- The RMSE of validation / testing score will be updated at `./results/Performance_Baby_plus_Pation_Lawn_and_Garden.txt`
- Iteration will be 300 epochs
- For this version, we remove early stopping to show the overall scores

### Citation

```
anonymous
```
