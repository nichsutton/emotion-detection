import matplotlib.pyplot as plt
import librosa
from numpy import size
import seaborn as sns
import librosa.display

def plot_emotions (data):
# Emotion training data visuals 
    plt.title("Emotion Samples", size=16)
    sns.countplot(data.Emotions)
    plt.xlabel("Sample Size", size=12)
    plt.ylabel("Emotion Type", size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()

def plot_wave (data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title("Audio Waveplots".format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    

