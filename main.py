from operator import index
import numpy as np
import pandas as pd
import warnings
import sys
from loaders import load_ravdess
from loaders import load_crema_d
from visualizers import librosa, plot_wave
import matplotlib.pyplot as plt
import seaborn as sns

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

ravdess = "./data/ravdess/"
crema_d = "./data/cremad/"

ravdess_data = load_ravdess(ravdess)
crema_d_data = load_crema_d(crema_d)

comb_data = pd.concat([ravdess_data, crema_d_data], axis=0)
comb_data.to_csv("comb_data_path.csv", index=False)

emotion = "sad"
path = np.array(comb_data.Path[comb_data.Emotions==emotion])[1]
data, sampling_rate = librosa.load(path)



