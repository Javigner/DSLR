import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    dataset = pd.read_csv(sys.argv[1], index_col = "Index")
    data = dataset[dataset.columns[6:19]]
    cols = data.columns.values
    
if __name__ == "__main__":
    main()