import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
      return(sigmoid(np.dot(X, theta)))

def cost(X, y, theta):
    predictions = sigmoid(X @ theta)
    error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
    error = np.array(error)
    print(error.shape)
    print(error.sum())
    return sum(error) / len(y);

def main():
    df = pd.read_csv('dataset_train.csv')
    #test = pd.read_csv('dataset_test.csv')
    y = df['Hogwarts House']
    X = np.array(df[df.columns[6:19]])
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    y = y.replace({'Slytherin': 1, 'Gryffindor': 2, 'Ravenclaw' : 1, 'Hufflepuff': 2})
    y = np.array(y)
    X = np.c_[np.ones(X.shape[0]), X]
    numFeatures = X.shape[1]
    numExamples = X.shape[0]
    numLabels = 4
    theta = np.zeros(numFeatures)
    print(cost(X, y, theta))
    
    
      
if __name__ == "__main__":
    main();