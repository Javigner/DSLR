import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

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
    return theta

def logistic_regression(X, y):
    y_grif = y.replace({'Slytherin': 0, 'Gryffindor': 1, 'Ravenclaw' : 0, 'Hufflepuff': 0})
    y_sly = y.replace({'Slytherin': 1, 'Gryffindor': 0, 'Ravenclaw' : 0, 'Hufflepuff': 0})
    y_rav = y.replace({'Slytherin': 0, 'Gryffindor': 0, 'Ravenclaw' : 1, 'Hufflepuff': 0})
    y_Huf = y.replace({'Slytherin': 0, 'Gryffindor': 0, 'Ravenclaw' : 0, 'Hufflepuff': 1})
    thetaG = All_vs_one_train(X, y_grif)
    thetaS = All_vs_one_train(X, y_sly)
    thetaR = All_vs_one_train(X, y_rav)
    thetaH = All_vs_one_train(X, y_Huf)
    return thetaG, thetaS, thetaR, thetaH
    
def main():
    if (len(sys.argv) < 2):
        sys.exit('Please give a valid Dataset')
    df = pd.read_csv(sys.argv[1])
    df = df.fillna(df.mean())
    y = df['Hogwarts House']
    #Preprocessing X
    df = df[df.columns[8:19]]
    df = df.drop('Care of Magical Creatures', axis=1)
    X = np.array(df)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = np.c_[np.ones(X.shape[0]), X]
    #X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    thetaG, thetaS, thetaR, thetaH = logistic_regression(X, y)
    theta = [thetaG, thetaS, thetaR, thetaH]
    theta = np.array(theta)
    np.savetxt("theta.csv", theta.T, delimiter=",", header="Gryffindor,Slytherin,Ravenclaw,Hufflepuf", comments="")    
    #logistic_regression(X_test, y_test)
    
if __name__ == "__main__":
    main();