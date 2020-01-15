import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns

def main():
    dataset = pd.read_csv(sys.argv[1], index_col = "Index")
    dataset.dropna(inplace=True)
    sns.pairplot(dataset, hue = 'Hogwarts House')
    plt.show()
    '''
        We can see that Arithmancy and Care of Magical Creatures are homogenius beetween houses and 
        Astronomy and Defense Against Dark Arts are identical so i will exclude them from my train_set
        
    '''
    
if __name__ == "__main__":
    main()