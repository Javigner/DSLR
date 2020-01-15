import pandas as pd
import matplotlib.pyplot as plt
import sys


def get_data(dataset, prep_dataset, house, col):
    df = prep_dataset[dataset["Hogwarts House"] == house][col]
    df.dropna(inplace=True)
    return df

def plot_hist(dataset, prep_dataset):
    for col in prep_dataset.columns:
        plt.figure()
        plt.hist(get_data(dataset, prep_dataset, "Gryffindor", col), bins=50, label = 'Gryffindor', color = 'r')
        plt.hist(get_data(dataset, prep_dataset, "Ravenclaw", col), bins=50,label = 'Ravenclaw', color = 'b')
        plt.hist(get_data(dataset, prep_dataset, "Slytherin", col), bins=50,label = 'Slytherin', color = 'g')
        plt.hist(get_data(dataset, prep_dataset, "Hufflepuff", col), bins=50,label = 'Hufflepuff', color = 'y')
        plt.legend(loc = 'upper right')
        plt.title(col)
        plt.show()
    
def main():
    dataset = pd.read_csv(sys.argv[1], index_col = "Index")
    data = dataset[dataset.columns[6:19]]
    cols = data.columns.values
    plot_hist(dataset, dataset[cols])
    
if __name__ == "__main__":
    main()