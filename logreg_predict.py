import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def StandardScaler(X):
    mean = np.mean(X, axis=0)
    scale = np.std(X - mean, axis=0)
    return (X - mean) / scale

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
      return(sigmoid(np.dot(X, theta)))
  
def main():
    if (len(sys.argv) < 3):
        sys.exit('Please give valid parameters')
    theta = pd.read_csv(sys.argv[2])
    df = pd.read_csv(sys.argv[1])
    df = df.fillna(df.mean())
    y = df['Hogwarts House']
    df = df[df.columns[8:19]]
    df = df.drop('Care of Magical Creatures', axis=1)
    X = np.array(df)
    X = StandardScaler(X)
    X = np.c_[np.ones(X.shape[0]), X]
    thetaG = theta['Gryffindor']
    thetaS = theta['Slytherin']
    thetaR = theta['Ravenclaw']
    thetaH = theta['Hufflepuf']
    predG = predict(X, thetaG)
    predS = predict(X, thetaS)
    predR = predict(X, thetaR)
    predH = predict(X, thetaH)
    result = []
    
    for i in range(len(predG)):
        Max = 0
        res = 0
        if (predG[i] > predS[i]):
            Max = predG[i]
            res = 'Gryffindor'
        else:
            Max = predS[i]
            res = 'Slytherin'
        if (predR[i] > Max):
            Max = predR[i]
            res = 'Ravenclaw'
        if (predH[i] > Max):
            Max = predH[i]
            res = 'Hufflepuff'
        result.append(res)
    y = y.tolist()
    somme = 0
    for z in range(len(result)):
        if (y[z] == result[z]):
            somme += 1
    print('\nAccuracy :', somme / len(result))
    result = np.array(result)
    result = pd.DataFrame(result, columns=['Hogwarts House'])
    result = result.rename_axis('Index', axis=1)
    result.to_csv('houses.csv')
if __name__ == "__main__":
    main();