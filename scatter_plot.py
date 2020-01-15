import pandas as pd
import matplotlib.pyplot as plt
import sys

def scatter_plot(dataset):
    plt.figure()
    plt.scatter(dataset['Astronomy'], dataset['Defense Against the Dark Arts'], label = 'students')
    plt.title("Features semblables")
    plt.xlabel("Astronomy")
    plt.ylabel("Defense Against the Dark Arts")
    plt.show()

def main():
    dataset = pd.read_csv(sys.argv[1], index_col = "Index")
    dataset.dropna(inplace=True)
    scatter_plot(dataset)
    
if __name__ == "__main__":
    main()