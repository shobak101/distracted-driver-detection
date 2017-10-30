import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit
import numpy as np

def myround(x, base=5):
    return int(base * round(float(x)/base))

# read data for c0 and c6
with open('c6.csv') as f:
    data = [list(line) for line in csv.reader(f)]

with open('c0.csv') as f:
    data = data + [list(line) for line in csv.reader(f)]

values = np.array(data)[:,1:3].astype(np.float)
target = np.array(data)[:,3].astype(np.float)

gnb = GaussianNB()

cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
print cross_val_score(gnb, values, target, cv=cv).mean()

