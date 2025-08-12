# 📊 Data Analysis & Visualization with Pandas, Matplotlib, and Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ==============================
# Task 1: Load and Explore Dataset
# ==============================
try:
    # Load Iris dataset from sklearn
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    df.rename(columns={'target': 'species'}, inplace=True)
    df['species'] = df['species'].map(dict(enumerate(iris_data.target_names)))

    # Display first few rows
    print("\nFirst 5 rows of dataset:")
    print(df.head())

    # Check data types & missing values
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Clean dataset (if missing values exist)
    df.dropna(inplace=True)

except FileNotFoundError:
    print("❌ Error: Dataset file not found.")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

# ==============================
# Task 2: Basic Data Analysis
# ==============================
# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Grouping by species and computing mean
grouped = df.groupby("species").mean()
print("\nAverage measurements by species:")
print(grouped)

# Identify patterns
print("\nPattern Insight:")
print("🌱 Setosa flowers generally have smaller petal sizes compared to Versicolor and Virginica.")

# ==============================
# Task 3: Data Visualization
# ==============================

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Line chart – Petal length over index (mock time series)
plt.figure(figsize=(8, 4))
plt.plot(df.index, df['petal length (cm)'], label="Petal Length")
plt.title("Petal Length Trend")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart – Average petal length per species
plt.figure(figsize=(8, 4))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'], palette="viridis")
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram – Distribution of sepal length
plt.figure(figsize=(8, 4))
sns.histplot(df['sepal length (cm)'], bins=15, kde=True, color="blue")
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot – Sepal length vs Petal length
plt.figure(figsize=(8, 4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette="deep")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
