                                        # ASSIGNMENT-5
import numpy as np
# Q.1 For the array gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]'), Find
# i. Sum of all elements
# ii. Sum of all elements row-wise
# iii. Sum of all elements column-wise
gfg = np.matrix([[4, 1, 9], [12, 3, 1], [4, 5, 6]])
sum_all = np.sum(gfg)
sum_row_wise = np.sum(gfg, axis=1)
sum_column_wise = np.sum(gfg, axis=0)
print("Sum of all elements:", sum_all)
print("Sum row-wise:\n", sum_row_wise)
print("Sum column-wise:\n", sum_column_wise)


# Q.2 (a)For the array: array = np.array([10, 52, 62, 16, 16, 54, 453]), find
# i. Sorted array
# ii. Indices of sorted array
# iii. 4 smallest elements
# iv. 5 largest elements
# (b) For the array: array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0]), find
# i. Integer elements only
# ii. Float elements only
array = np.array([10, 52, 62, 16, 16, 54, 453])
sorted_array = np.sort(array)
sorted_indices = np.argsort(array)
smallest_4 = np.sort(array)[:4]
largest_5 = np.sort(array)[-5:]
print("Sorted array:", sorted_array)
print("Indices of sorted array:", sorted_indices)
print("4 smallest elements:", smallest_4)
print("5 largest elements:", largest_5)

array_b = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
integer_elements = array_b[array_b == array_b.astype(int)]
float_elements = array_b[array_b != array_b.astype(int)]
print("Integer elements:", integer_elements)
print("Float elements:", float_elements)


# Q.3 You are given a weekly sales dataset and need to perform various operations
# using NumPy broadcasting.
# a) Generate your unique sales dataset:
#  Take the sum of the ASCII values of the initials of your first and last
# name. Call this value X. (If your initials are A B → ASCII sum = 65
# + 66 = 131 → sales = [131, 181, 231, 281, 331].)
#  Create a NumPy array sales with values [X, X+50, X+100, X+150,
# X+200].
# b) Compute your personalized tax rate as ((X % 5) + 5) / 100.
#  Use broadcasting to apply this tax rate to each sales value.
# c) Adjust sales based on discount:
#  If sales < X+100, apply a 5% discount.
#  If sales >= X+100, apply a 10% discount.
# d) Expand sales data for multiple weeks:
#  Create a 3×5 matrix representing three weeks of sales by stacking
# sales three times using broadcasting.
#  Increase sales by 2% per week using element-wise broadcasting.
initials = "AB"
X = sum(ord(char) for char in initials)
sales = np.array([X, X + 50, X + 100, X + 150, X + 200])
tax_rate = ((X % 5) + 5) / 100
taxed_sales = sales * (1 + tax_rate)
discounted_sales = np.where(sales < X + 100, sales * 0.95, sales * 0.90)
weekly_sales = np.tile(sales, (3, 1)) 
weekly_sales = weekly_sales * np.array([1, 1.02, 1.04]).reshape(3, 1)
print("Sales Data:", sales)
print("Personalized Tax Rate:", tax_rate)
print("Taxed Sales:", taxed_sales)
print("Discounted Sales:", discounted_sales)
print("Weekly Sales Matrix:\n", weekly_sales)


# Q4. Generate x values using np.linspace() from -10 to 10 with 100 points. Use
# each function from the list below and compute y values using NumPy:
#  Y = x2
#  Y = sin(x)
#  Y = ex
#  Y = log(|x| + 1)
# Plot the chosen function using Matplotlib. Add title, labels, and grid for clarity. 
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)
y_square = x**2
y_sin = np.sin(x)
y_exp = np.exp(x)
y_log = np.log(np.abs(x) + 1)

# Plotting Y = x^2
plt.figure(figsize=(8, 5))
plt.plot(x, y_square, label='y = x^2', color='blue')
plt.title('Plot of y = x²')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()

# Plotting Y = sin(x)
plt.figure(figsize=(8, 5))
plt.plot(x, y_sin, label='y = sin(x)', color='green')
plt.title('Plot of y = sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()

# Plotting Y = e^x
plt.figure(figsize=(8, 5))
plt.plot(x, y_exp, label='y = e^x', color='red')
plt.title('Plot of y = e^x')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([-10, 10000])  
plt.grid()
plt.legend()
plt.show()

# Plotting Y = log(|x| + 1)
plt.figure(figsize=(8, 5))
plt.plot(x, y_log, label='y = log(|x| + 1)', color='purple')
plt.title('Plot of y = log(|x| + 1)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
