import numpy as np
from sklearn.linear_model import SGDClassifier 
import pandas as pd
import sys

def StandardScaler(X):
    mean = np.mean(X, axis=0)
    scale = np.std(X - mean, axis=0)
    return (X - mean) / scale

def Preprocessing(df):
    df = df.fillna(df.mean())
    y = df['Hogwarts House']
    df = df[df.columns[8:19]]
    df = df.drop('Care of Magical Creatures', axis=1)
    y = y.replace({'Slytherin': 0, 'Gryffindor': 1, 'Ravenclaw' : 2, 'Hufflepuff': 3})
    X = np.array(df)
    X = StandardScaler(X)
    X = np.c_[np.ones(X.shape[0]), X]
    return df, X, y

def main():
    if (len(sys.argv) < 3):
            sys.exit('Please give a valid Dataset')
    df = pd.read_csv(sys.argv[1])
    df, X, y = Preprocessing(df)
    #Preprocessing X
    model = SGDClassifier(max_iter=1000, eta0=0.001, loss='log')
    model.fit(X, y)
    print('Train Accuracy score :', model.score(X, y)) 
    df_test = pd.read_csv(sys.argv[2])
    df, X, y = Preprocessing(df_test)
    print('Test Accuracy score :', model.score(X, y))
    
if __name__ == "__main__":
    main();