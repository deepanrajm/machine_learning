# Create your first MLP in Keras

import numpy as np
import seaborn as sns
from pandas import read_csv
import matplotlib.pyplot as plt
sns.set()


dataframe = read_csv("iris.csv")


print(dataframe.head())


ax = sns.scatterplot(x="petal.length", y="petal.width",hue="variety",size = "sepal.width",data=dataframe)
plt.show()