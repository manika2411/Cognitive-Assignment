                                #ASSIGNMENT-6
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Q1: Plotting mathematical functions
M = float(input("Enter the value of M: "))
x = np.linspace(-10, 10, 100)
y1 = M * (x ** 2)
y2 = M * np.sin(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='y = M*x^2', color='blue', linestyle='--')
plt.plot(x, y2, label='y = M*sin(x)', color='red', linestyle='-')
plt.title('Mathematical Functions')
plt.xlabel('x values')
plt.ylabel('y values')
plt.legend()
plt.grid()
plt.show()


# Q2: Creating a dataset and plotting a bar graph
subjects = ['Math', 'Science', 'English', 'History', 'Art']
scores = [85, 78, 90, 72, 88]
df = pd.DataFrame({'Subjects': subjects, 'Scores': scores})

plt.figure(figsize=(8, 5))
ax = sns.barplot(x='Subjects', y='Scores', data=df, palette='viridis')
for index, value in enumerate(scores):
    plt.text(index, value + 1, str(value), ha='center', fontsize=12)
plt.title('Student Scores')
plt.xlabel('Subjects')
plt.ylabel('Scores')
plt.grid(axis='y')
plt.show()


# Q3: Generating random dataset with roll number as seed
roll_number = int(input("Enter your roll number: "))
np.random.seed(roll_number)
data = np.random.randn(50)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].plot(np.cumsum(data), color='blue')
axs[0, 0].set_title('Cumulative Sum')
axs[0, 0].set_xlabel('Index')
axs[0, 0].set_ylabel('Sum')
axs[0, 0].grid()

axs[1, 1].scatter(range(50), data, color='red')
axs[1, 1].set_title('Scatter Plot with Random Noise')
axs[1, 1].set_xlabel('Index')
axs[1, 1].set_ylabel('Value')
axs[1, 1].grid()

fig.delaxes(axs[0, 1])
fig.delaxes(axs[1, 0])
plt.tight_layout()
plt.show()


# Q4: Analyzing Company Sales Data
dataset_url = "https://raw.githubusercontent.com/AnjulaMehto/MCA/main/company_sales_data.csv"
df_sales = pd.read_csv(dataset_url)

# 1. Line plot of total profit using seaborn
plt.figure(figsize=(8, 5))
sns.lineplot(x='month_number', y='total_profit', data=df_sales, marker='o', color='blue')
plt.title('Total Profit per Month')
plt.xlabel('Month')
plt.ylabel('Total Profit')
plt.grid()
plt.show()

# 2. Multi-line plot of all product sales
plt.figure(figsize=(10, 6))
for product in df_sales.columns[1:6]:
    sns.lineplot(x='month_number', y=product, data=df_sales, label=product)
plt.title('Product Sales per Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()

# 3. Bar plot for all features/attributes
df_melted = df_sales.melt(id_vars=['month_number'], var_name='Product', value_name='Sales')
plt.figure(figsize=(14, 6))
sns.barplot(x='month_number', y='Sales', hue='Product', data=df_melted)
plt.title('Monthly Sales Overview')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid()
plt.show()
