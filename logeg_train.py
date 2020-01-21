import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 

df = pd.read_csv('dataset_train.csv')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
      return(sigmoid(np.dot(X, theta)))

def cost(X, y, theta):
    return((-1 / X.shape[0]) * np.sum(y * np.log(predict(X, theta)) + (1 - y) * np.log(1 - predict(X, theta))))

def fit(X, y, theta, alpha, num_iters, m):
    tmp = theta
    for _ in tqdm(range(num_iters)):
        theta = theta - alpha * (1.0 / len(X)) * (np.dot((predict(X, theta) - y), X))
        if tmp.all() == theta.all():
            break;
    return theta

def All_vs_one_train(X, y):
    y = np.array(y)
    theta = np.zeros(X.shape[1])
    theta = fit(X, y, theta, 2, 6000, X.shape[0])
    pred = predict(X, theta)
    return theta, pred
    
def main():
    df = pd.read_csv('dataset_train.csv')
    df = df.dropna()
    y = df['Hogwarts House']
   
    #Preprocessing X
    X = np.array(df[df.columns[6:19]])
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = np.c_[np.ones(X.shape[0]), X]
    y_grif = y.replace({'Slytherin': 0, 'Gryffindor': 1, 'Ravenclaw' : 0, 'Hufflepuff': 0})
    y_sly = y.replace({'Slytherin': 1, 'Gryffindor': 0, 'Ravenclaw' : 0, 'Hufflepuff': 0})
    y_rav = y.replace({'Slytherin': 0, 'Gryffindor': 0, 'Ravenclaw' : 1, 'Hufflepuff': 0})
    y_Huf = y.replace({'Slytherin': 0, 'Gryffindor': 0, 'Ravenclaw' : 0, 'Hufflepuff': 1})
    thetaG, predG = All_vs_one_train(X, y_grif)
    thetaS, predS = All_vs_one_train(X, y_sly)
    thetaR, predR = All_vs_one_train(X, y_rav)
    thetaH, predH = All_vs_one_train(X, y_Huf)
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
    y = df['Hogwarts House'].tolist()
    somme = 0
    for z in range(len(result)):
        if (y[z] == result[z]):
            somme += 1
    print(somme / len(result))
            
    
if __name__ == "__main__":
    main();