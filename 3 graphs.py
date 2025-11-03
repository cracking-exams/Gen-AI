import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Histogram of petal length
plt.figure(figsize=(6,4))
sns.histplot(iris['petal_length'], kde=True, color='skyblue')
plt.title('Distribution of Petal Length')
plt.show()

# Scatter plot: sepal length vs sepal width
plt.figure(figsize=(6,4))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(iris.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
