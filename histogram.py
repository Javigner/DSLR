import pandas as pd
import matplotlib.pyplot as plt
import sys


def preprocess(dataset):
    data = dataset[dataset.columns[6:19]]
    cols = data.columns.values
    for col in cols:
        dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()
    return dataset[cols]

def get_grades(dataset, prep_dataset, house, topic):
    df = prep_dataset[dataset["Hogwarts House"] == house][topic]
    df.dropna(inplace=True)
    return df

def plot_hist(dataset, prep_dataset):
    for col in prep_dataset.columns:
        plt.figure()
        plt.hist(get_grades(dataset, prep_dataset, "Gryffindor", col), bins=10, alpha=0.5, label = 'Gryffindor', color = 'r')
        plt.hist(get_grades(dataset, prep_dataset, "Ravenclaw", col), bins=10, alpha=0.5, label = 'Ravenclaw', color = 'b')
        plt.hist(get_grades(dataset, prep_dataset, "Slytherin", col), bins=10, alpha=0.5, label = 'Slytherin', color = 'g')
        plt.hist(get_grades(dataset, prep_dataset, "Hufflepuff", col), bins=10, alpha=0.5, label = 'Hufflepuff', color = 'y')
        plt.legend(loc = 'upper right')
        plt.title(col)
        plt.show()
    
if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col = "Index")
    prep_dataset = preprocess(dataset)
    plot_hist(dataset, prep_dataset)