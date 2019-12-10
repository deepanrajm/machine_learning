import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


data = pd.read_csv('mobile_cleaned.csv')
data.head()

ax = sns.scatterplot(x="stand_by_time", y="battery_capacity", data=data)
plt.show()

ax = sns.scatterplot(x = "stand_by_time", y = "battery_capacity", hue="thickness", data=data)
plt.show()

ax = sns.distplot(data["stand_by_time"])
plt.show()


ax = sns.boxplot(x="is_liked", y="battery_capacity", data=data)
plt.show()

ax = sns.boxplot(x = "expandable_memory", y = "price", data=data)
plt.show()