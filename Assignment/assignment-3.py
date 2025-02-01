                                    # ASSIGNMENT-3   
import pandas as pd
# 1. Create a dataset as follow in the table.
data = {
    'Tid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Refund': ["Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No"],
    'Marital Status': ["Single", "Married", "Single", "Married", "Divorced", "Married", "Divorced", "Single", "Married", "Single"],
    'Taxable Income': ["125K", "100K", "70K", "120K", "95K", "60K", "220K", "85K", "75K", "90K"],
    'Cheat': ["No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes"]
}
df=pd.DataFrame(data)


# 2. From the above table that you have created, locate row 0, 4, 7 and 8 using DataFrame.
print(df.loc[0],"\n\n",df.loc[4],"\n\n",df.iloc[7:9])


# 3 Navigate the DataFrame and do the following task for the table created in question 1:
# 1. Select row from index 3 to 7.
# 2. Select row from index 4 to 8, and column 2 to 4.
# 3. Select all rows with column index 1 to 3 (include index 3 during selection).
print("\n\n",df.loc[3:7])
print("\n\n",df.iloc[4:9,2:5])
print("\n\n",df.iloc[:,1:4])


# 4. Read a csv file and display its first five rows.
df=pd.read_csv('Iris.csv')
print(df.head)


# 5.From the csv file (uploaded in the Q.4) delete row 4, and delete column 3. Display the
# result.
df.drop(1,axis=0,inplace=True)
df.drop(df.columns[1],axis=1,inplace=True)
print(df.head)


# 6 Create a sample dataset (employees.csv) containing information about employees in
# a company.
df=pd.read_csv('dataset.csv')
print(df.head)
# a) Shape (number of rows and columns) of the DataFrame
print("shape of data frame: ",df.shape)

# b) Summary of the DataFrame that includes the data types and non-null counts for
# each column.
print("summary of data frame: ",df.info())

# c) Generate descriptive statistics.
print("desription of data frame :",df.describe())

# d) Display first 5 rows and last 3 rows
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 3 rows:")
print(df.tail(3))

# e) Calculate specific statistics
print("Average Salary:", df["Salary"].mean())
print("Total Bonus Paid:", df["Bonus"].sum())
print("Youngest Employee's Age:", df["Age"].min())
print("Highest Performance Rating:", df["Rating"].max())

# f) Sort DataFrame by Salary in descending order
sorted_df = df.sort_values(by="Salary", ascending=False)
print("\nDataFrame sorted by Salary (descending):")
print(sorted_df)

# g) Add Performance Category column
def categorize_performance(rating):
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 4.0:
        return "Good"
    else:
        return "Average"
df["Performance_Category"] = df["Rating"].apply(categorize_performance)
print("DataFrame with Performance Category:")
print(df)

# h) Identify missing values
print("Missing values in DataFrame:")
print(df.isnull().sum())

# i) Rename Employee_ID column to ID
df.rename(columns={"Employee_ID": "ID"}, inplace=True)
print("\nDataFrame with renamed column:")
print(df.head())

# j) Find employees with more than 5 years experience and in IT department
exp_df = df[df["Years_of_Experience"] > 5]
it_df = df[df["Department"] == "IT"]
print("Employees with more than 5 years of experience:")
print(exp_df)
print("Employees in IT department:")
print(it_df)

# k) Add Tax column (10% deduction from Salary)
df["Tax"] = df["Salary"] * 0.10
print("\nDataFrame with Tax column:")
print(df)

# l) Save modified DataFrame to a new CSV file
df.to_csv("employees_modified.csv", index=False)
print("\nModified dataset saved as employees_modified.csv")