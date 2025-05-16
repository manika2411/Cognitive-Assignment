import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Part I: Randomized Sales Data Generation
# Step 1: Initialize Random Seed
roll_number = 123456  # Replace with your actual roll number
np.random.seed(roll_number)

# Step 2: Generate Sales Data
sales_data = np.random.randint(1000, 5001, size=(12, 4))

# Step 3: Convert to DataFrame
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
categories = ["Electronics", "Clothing", "Home & Kitchen", "Sports"]
df = pd.DataFrame(sales_data, columns=categories, index=months)

# Part II: Data Manipulation and Analysis
# Display first 5 rows and summary statistics
print(df.head())
print(df.describe())

# Total sales per category
total_sales_category = df.sum()
print("Total Sales per Category:\n", total_sales_category)

# Total sales per month
df["Total Sales"] = df.sum(axis=1)
print("Total Sales per Month:\n", df["Total Sales"])

# Average sales growth between consecutive months
df["Growth Rate"] = df["Total Sales"].pct_change() * 100

# Apply discount based on roll number
if roll_number % 2 == 0:
    df["Electronics"] *= 0.9  
else:
    df["Clothing"] *= 0.85 

# Part III: Visualizations
plt.figure(figsize=(10, 6))
for category in categories:
    plt.plot(df.index, df[category], marker='o', label=category)
plt.xlabel("Months")
plt.ylabel("Sales")
plt.title("Monthly Sales Trends")
plt.legend()
plt.grid()
plt.show()

# Box plots for sales distribution
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[categories])
plt.title("Sales Distribution per Category")
plt.show()


# Q2: NumPy Array Operations
array = np.array([[1, -2, 3], [-4, 5, -6]])

# Element-wise absolute value
abs_values = np.abs(array)
print(abs_values)

# Percentiles
percentiles = {p: np.percentile(array, p, axis=None) for p in [25, 50, 75]}
col_percentiles = {p: np.percentile(array, p, axis=0) for p in [25, 50, 75]}
row_percentiles = {p: np.percentile(array, p, axis=1) for p in [25, 50, 75]}
print(percentiles)
print(col_percentiles)
print(row_percentiles)

# Mean, Median, and Standard Deviation
mean_val = np.mean(array)
median_val = np.median(array)
std_dev = np.std(array)
print(mean_val)
print(median_val)
print(std_dev)


# Q3: NumPy Floor, Ceiling, and Rounding
arr = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])
floor_vals = np.floor(arr)
ceil_vals = np.ceil(arr)
trunc_vals = np.trunc(arr)
round_vals = np.round(arr)
print(floor_vals)
print(ceil_vals)
print(trunc_vals)
print(round_vals)


# Q4: Swap Two Elements in a List Using a Temporary Variable
lst = [1, 2, 3, 4, 5]
i, j = 1, 3 
temp = lst[i]
lst[i] = lst[j]
lst[j] = temp
print("List after swapping:", lst)


# Q5: Swap Two Elements in a Set by Converting to a List
s = {10, 20, 30, 40, 50}
list_s = list(s)
i, j = 1, 3 
list_s[i], list_s[j] = list_s[j], list_s[i]
s = set(list_s)
print("Set after swapping:", s)
