import pandas as pd
from loaders import load_ravdess, load_crema_d
import matplotlib.pyplot as plt
import seaborn as sns
from extractors import get_feautures

ravdess = "./data/ravdess/"
crema_d = "./data/cremad/"

ravdess_data = load_ravdess(ravdess)
crema_d_data = load_crema_d(crema_d)

data_path = pd.concat([ravdess_data, crema_d_data], axis=0)
sample_rate = 22050

X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
    try:
        feature = get_feautures(path, sample_rate)
        for ele in feature:
            X.append(ele)
            Y.append(emotion)
    except TypeError as identifier:
        pass

Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index=False)


# Plots the count of emotions
plt.title('Count of Emotions', size=16)
sns.countplot(data_path.Emotions)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()
