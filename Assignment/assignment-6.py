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

plt.figure(figsize=(8, 5))
plt.plot(df_sales['month_number'], df_sales['total_profit'], marker='o', linestyle='-', color='blue')
plt.title('Total Profit per Month')
plt.xlabel('Month')
plt.ylabel('Total Profit')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
for column in df_sales.columns[1:6]:  
    plt.plot(df_sales['month_number'], df_sales[column], label=column)
plt.title('Product Sales per Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
df_sales.set_index('month_number').plot(kind='bar', figsize=(12, 6))
plt.title('Sales Data Overview')
plt.xlabel('Month')
plt.ylabel('Values')
plt.grid()
plt.show()
