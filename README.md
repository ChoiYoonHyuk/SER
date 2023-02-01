## [CIKM '22] Review-Based Domain Disentanglement without Duplicate Users or Contexts for Cross-Domain Recommendation

### Project Structure

```
.
├── README.md
├── requirements.txt
├── resources
│   ├── Musical_Instruments.json
│   └── Patio_Lawn_and_Garden.json
└── SER
    ├── main.py
    ├── model.py
    └── preprocess.py
```

### Setup

- Setup Conda, PyTorch, CUDA
  - > pip install -r requirements.txt
- Download Pretrained Word Embeddings; GloVe (below)
  - https://nlp.stanford.edu/data/glove.6B.zip
  - Place under `'./resources/glove.6B/glove.6B.100d.txt'`
- Source & Target domain dataset
  - can be excuted with additional downloads: `http://jmcauley.ucsd.edu/data/amazon/`
  - substitute
    - `Baby.json` > `Musical_Instruments.json`
    - `Musical_Instruments.json`, `Pation_Lawn_and_Garden.json`

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
https://dl.acm.org/doi/abs/10.1145/3511808.3557434
```
