# Environment
    # Conda, PyTorch
    # We freeze the installed modules as requirements.txt

# Files
    # Source & Target domain data (can be excuted with additional downloads: http://jmcauley.ucsd.edu/data/amazon/)
        # GitHub file size limit (< 25 MB)
        # We substitute Baby.json > Musical_Instruments.json
        # Musical_Instruments.json, Pation_Lawn_and_Garden.json

    # DECRec.py
        # Main file, load the files below

    # DECRec_Process_Data.py
        # Pre-process source & target domain data

    # DECRec_Module.py
        # Training & Inference with PyTorch

# Execution
    # Download pretrained word embeddings, GloVe (below)
        # https://nlp.stanford.edu/data/glove.6B.zip
        # Unzipped file name should be './glove.6B.100d.txt'
    # Training / Test with following code
    # python DECRec.py

# Performance
    # The RMSE of validation / testing score will be updated at Performance_Baby_plus_Pation_Lawn_and_Garden.txt
    # Iteration will be 300 epochs
    # For this version, we remove early stopping to show the overall scores
