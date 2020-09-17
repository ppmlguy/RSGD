import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def majority_label(y):
    df = pd.DataFrame(y)

    counts = df.iloc[:, 0].value_counts().values

    return (np.max(counts)*1.0)/np.sum(counts)


def load_dat(filepath, minmax=(0, 1), shuffle=False, normalize=-1, verbose=False):
    df = pd.read_csv(filepath, dtype=np.float32, delim_whitespace=True)

    if shuffle:
        df = df.sample(frac=1.0)

    lines = df.values
    if verbose:
        print("{} ({} rows)".format(filepath, lines.shape[0]))

    labels = lines[:, -1]
    features = lines[:, :-1]

    N, dim = features.shape

    minmax = MinMaxScaler(feature_range=minmax, copy=False)
    minmax.fit_transform(features)

    if normalize > 0:
        xnorm = np.linalg.norm(features, axis=1)
        if verbose:
            print("{} rows are clipped.".format(np.count_nonzero(xnorm > normalize)))
        features /= np.maximum(1, (xnorm[:, np.newaxis]/normalize))

    X = np.hstack([np.ones(shape=(N, 1)), features])

    return X, labels

