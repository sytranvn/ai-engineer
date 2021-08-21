import pandas as pd
import plot_data
data = pd.read_csv('./ex1data1.txt', header=None)
X = data[0]
y = data[1]
m = len(X)

plot_data(X, y)
