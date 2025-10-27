# ============================================
# 💡 NumPy + Pandas + Visualization Reference Script
# Frequently Used Functions in ML & Data Science
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# 1️⃣ NUMPY BASICS
# --------------------------------------------

# Creating Arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Array creation helpers
zeros = np.zeros((2, 3))
ones = np.ones((3, 3))
arange_arr = np.arange(0, 10, 2)
linspace_arr = np.linspace(0, 1, 5)
rand_arr = np.random.rand(3, 3)
rand_int = np.random.randint(0, 100, (3, 3))

# Array operations
sum_arr = np.sum(matrix)
mean_arr = np.mean(matrix)
std_arr = np.std(matrix)
max_arr = np.max(matrix)
min_arr = np.min(matrix)
argmax_arr = np.argmax(matrix)
argmin_arr = np.argmin(matrix)

# Elementwise operations
added = arr + 5
squared = arr ** 2
sqrt_arr = np.sqrt(arr)

# Boolean indexing
mask = arr > 2
filtered = arr[mask]

# Matrix operations
dot_product = np.dot(matrix, matrix.T)
transposed = matrix.T
flattened = matrix.flatten()

# Random seed reproducibility
np.random.seed(42)
rand_normal = np.random.normal(0, 1, (3, 3))

# --------------------------------------------
# 2️⃣ PANDAS BASICS
# --------------------------------------------

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, np.nan, 28],
    'Salary': [50000, 60000, 75000, 80000, np.nan],
    'Department': ['HR', 'Tech', 'Tech', 'HR', 'Finance']
}
df = pd.DataFrame(data)
print("\nInitial DataFrame:\n", df)

# Basic info
print("\n--- DataFrame Info ---")
print(df.info())
print("\nDescribe numerical columns:\n", df.describe())

# --------------------------------------------
# 3️⃣ DATA CLEANING & TRANSFORMATION
# --------------------------------------------

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)
df.rename(columns={'Salary': 'Monthly_Income'}, inplace=True)
df['Age'] = df['Age'].astype(int)
df['Experience_Years'] = [2, 5, 7, 10, 3]
df['Tax'] = df['Monthly_Income'].apply(lambda x: 0.1 * x)
df['Net_Income'] = df.apply(lambda r: r['Monthly_Income'] - r['Tax'], axis=1)

# --------------------------------------------
# 4️⃣ MERGE / CONCAT / GROUPBY
# --------------------------------------------

dept_data = pd.DataFrame({
    'Department': ['HR', 'Tech', 'Finance'],
    'Manager': ['Sophia', 'Liam', 'Noah']
})
merged_df = pd.merge(df, dept_data, on='Department', how='left')

extra = pd.DataFrame({
    'Name': ['Frank'],
    'Age': [26],
    'Monthly_Income': [48000],
    'Department': ['Tech'],
    'Experience_Years': [1],
    'Tax': [4800],
    'Net_Income': [43200],
    'Manager': ['Liam']
})
concat_df = pd.concat([merged_df, extra], ignore_index=True)
concat_df.set_index('Name', inplace=True)

# --------------------------------------------
# 5️⃣ ANALYSIS
# --------------------------------------------

print("\n--- Descriptive Stats ---")
print(concat_df.describe())

print("\nCorrelation Matrix:\n", concat_df.corr(numeric_only=True))
print("\nDepartment Counts:\n", concat_df['Department'].value_counts())

# --------------------------------------------
# 6️⃣ VISUALIZATION
# --------------------------------------------

plt.figure(figsize=(10, 5))
sns.barplot(x='Department', y='Monthly_Income', data=concat_df, estimator=np.mean)
plt.title('Average Monthly Income by Department')
plt.xlabel('Department')
plt.ylabel('Average Income')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(concat_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(concat_df['Age'], kde=True, bins=5)
plt.title('Age Distribution')
plt.tight_layout()
plt.show()

# --------------------------------------------
# 7️⃣ EXPORT & IMPORT
# --------------------------------------------

concat_df.to_csv('example_data.csv')
print("\n✅ Data saved to 'example_data.csv'")

loaded_df = pd.read_csv('example_data.csv')
print("\nLoaded CSV Preview:\n", loaded_df.head())

print("\n🎯 Script executed successfully — NumPy + Pandas + Visualization refresher complete!")
