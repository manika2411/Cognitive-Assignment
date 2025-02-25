import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
M=float(input("Enter value of M: "))
x=np.linspace(-10,10,100)
y1= M*x*x
y2=M*np.sin(x)
plt.plot(x,y1,'r--',label='Y=M*x^2')
plt.plot(x,y2,'b-',label='Y=M*sin(x)')
plt.legend()
plt.grid()
plt.title('Mathematical Functions')
plt.show


import pandas as pd
import matplotlib.pyplot as plt
# reading the database
data = pd.read_csv("subjects_scores.csv")
colors = ['blue', 'green', 'red', 'yellow','black']
# Scatter plot with day against tip
plt.bar(data['Subject'], data['Score'],color=colors)
# Adding Title to the Plot
plt.title("Scatter Plot")
# Setting the X and Y labels
plt.xlabel('Subject')
plt.ylabel('Score')
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Set the seed using your roll number (e.g., 12345)
roll_number = 102497021  # Replace with your actual roll number
np.random.seed(roll_number)

# Generate a dataset of 50 values using np.random.randn()
data = np.random.randn(50)

# Create a 2x2 subplot layout
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Line plot showing the cumulative sum
axs[0].plot(np.cumsum(data), label='Cumulative Sum', color='blue')
axs[0].set_title('Cumulative Sum Plot')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Cumulative Sum')
axs[0].grid(True)
axs[0].legend()

# Scatter plot with random noise
axs[1].scatter(np.arange(50), data + np.random.randn(50) * 0.5, label='Data with Noise', color='red')
axs[1].set_title('Scatter Plot with Random Noise')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Value')
axs[1].grid(True)
axs[1].legend()

# Adjust the layout
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the URL
url = "https://raw.githubusercontent.com/AnjulaMehto/MCA/main/company_sales_data.csv"
df = pd.read_csv(url)

# Print the first few rows and columns of the dataset to ensure correct structure
print(df.head())
print(df.columns)

# Set the Seaborn style for better visuals
sns.set(style="whitegrid")

# 1. Line plot for Total Profit over months
plt.figure(figsize=(10, 6))
sns.lineplot(x=df['month_number'], y=df['total_profit'], marker='o', color='b')
plt.title('Total Profit Over Months')
plt.xlabel('Month Number')
plt.ylabel('Total Profit')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Multiline plot for Sales of all products over months
# Identifying the columns for product sales
product_columns = ['facecream', 'facewash', 'toothpaste', 'bathingsoap', 'shampoo', 'moisturizer']

plt.figure(figsize=(10, 6))
for col in product_columns:
    sns.lineplot(x=df['month_number'], y=df[col], label=col)
plt.title('Sales of All Products Over Months')
plt.xlabel('Month Number')
plt.ylabel('Sales')
plt.legend(title="Products", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Bar chart for all features (attributes) in the dataset
plt.figure(figsize=(12, 8))
df.drop(columns='month_number').mean().plot(kind='bar', color='c')
plt.title('Average Value for Each Feature')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()