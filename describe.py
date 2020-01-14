import pandas as pd
import numpy as np
from math import sqrt
import sys


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
        Q1.append(col[int(len(col) * 0.25)])
        Q2.append(col[int(len(col) * 0.5)])
        Q3.append(col[int(len(col) * 0.75)])
    desc = list(zip(count, mean, std, lmin, Q1, Q2, Q3, lmax))
    column = data.columns.values
    desc = np.transpose(desc)
    df = pd.DataFrame(desc, columns =column, index=['count','mean','std ','min', '25%', '50%', '75%', 'max'])
    print(df)
    
def main():
    if (len(sys.argv) < 2):
        sys.exit('Please give a valid Dataset')
    df_train = pd.read_csv(sys.argv[1])
    describe_data(df_train)
    
if __name__ == "__main__":
    main();