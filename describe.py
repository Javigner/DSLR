import pandas as pd
import numpy as np
from math import sqrt

def describe_data(data):
    data = data[data.columns[6:19]]
    count = []
    mean = []
    std = []
    lmin = []
    lmax = []
    Q1 = []
    Q2 = []
    Q3 = []
    for i in range(data.shape[1]):
        col = data.iloc[:, i].tolist()
        col = [x for x in col if str(x) != 'nan']
        col.sort()
        count.append(len(col))
        mean.append(sum(col) / len(col))
        result = 0
        for x in range(len(col)):
            result += (col[x] - sum(col) / len(col))**2 
        result = result / (len(col) - 1)
        std.append(sqrt(result))
        lmin.append(min(col))
        lmax.append(max(col))
        test = (len(col) - 1) * 0.5
        Q1.append(col[int(len(col) * 0.25)])
        Q2.append(col[int(len(col) * 0.5)])
        Q3.append(col[int(len(col) * 0.75)])
    print(Q2)
            
    
def main():
    df_test = pd.read_csv('dataset_test.csv')
    df_train = pd.read_csv('dataset_train.csv')
    print(df_train.describe())
    describe_data(df_train)
    
if __name__ == "__main__":
    main();