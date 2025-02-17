from pathlib import Path
import pickle
import numpy as np
import functools
import os
import requests


this_dir, this_filename = os.path.split(__file__) 


def get_model(remote_path):
    # download the model parameters
    res = requests.get(remote_path)
    with open(Path(this_dir) / 'models' / 'ace_clf.pkl', 'wb') as f:
        f.write(res.content)

@functools.lru_cache(maxsize=None)
def load_model():
    with open(Path(this_dir)/ 'models' /'ace_clf.pkl','rb') as f:
        clf = pickle.load(f)

    return clf