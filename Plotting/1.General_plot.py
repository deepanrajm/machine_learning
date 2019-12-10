import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


x = np.random.random((10, 1))

print(x)

plt.plot(x, '*-')
plt.show()

x = np.linspace(0, 100, 50)
y = np.power(x, .5)
plt.plot(x, y, '+-')
plt.show()

sns.set()
sns.lineplot(x, y)
plt.show()

uniform_data = np.random.rand(10, 12)
print(uniform_data)
ax = sns.heatmap(uniform_data, cmap="YlGnBu")
plt.show()
