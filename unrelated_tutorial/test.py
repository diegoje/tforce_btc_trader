import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def one_hot_encoder(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


df = pd.read_csv("/Volumes/SD64GB/GoogleDrive/tforce_btc_trader/unrelated_tutorial/data.csv")

X = df[df.columns[0:60]].values
y = df[df.columns[60]]


# encode the dependent variable
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encoder(y)





