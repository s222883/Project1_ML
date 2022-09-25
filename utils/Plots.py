import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Donutplot:
    def __init__(self, sizes, colors, circle_ratio):
        self.colors = colors
        self.sizes = sizes
        self.circle_ratio = circle_ratio
        # self.labels = labels
    def plot(self):
        plt.pie(self.sizes, colors = self.colors,radius = 1.1,
                wedgeprops = {'linewidth': 1, 'edgecolor': 'white'},
                autopct='%1.1f%%',textprops = dict(color = "w",size = 10,weight = 'bold'))
        my_circle = plt.Circle((0, 0), self.circle_ratio, color = 'white')
        p = plt.gcf()
        p.gca().add_artist(my_circle)

if __name__ == '__main__':
    df = pd.read_csv(r'/Users/diego/Documents/UNIOVI/TFG/DATA/Breast_cancer.csv')
    n_benign = sum(df.diagnosis == 'B')  # Número de nódulos benignos
    n_malign = sum(df.diagnosis == 'M')  # Número de nódulos benignos
    color = {"granate": "#BA4A00",
             "amarillo": "#F5B041",
             "verde": "#148F77",
             "blue": "#0051A2",
             "red": "#DD1717"}
    size = [n_benign,n_malign]
    labs =  np.around([100*n_benign/sum(size),100*n_malign/sum(size)],2)
    d = Donutplot(sizes=size, colors=(color["blue"], color["red"]), circle_ratio=0.4)
    d.plot()
    plt.legend(['Benigno', 'Maligno'], title='Tipo de nódulo', bbox_to_anchor=(0.95, 1))
    # plt.savefig('/Users/diego/Documents/UNIOVI/TFG/LATEX/Media/proporcion.jpg', dpi = 500)

    plt.show()