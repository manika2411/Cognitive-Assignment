                                     #ASSIGNMENT-1
# 1: WAP to print your name three times
i=1
while(i<=3):
    print("Manika Singla")
    i=i+1


# 2: WAP to add three numbers and print the result
a=7
b=2
c=9
print(a+b+c)


# 3: WAP to concatinate three strings and print the result.
a="Manika"
b="Singla"
c="2Q14"
print(a+" "+b+" "+c)


# 4: WAP to print the table of 7, 9.
t=7
for i  in range(1,11):
    print(7,"*",i,"=",7*i)
t=9   
for i  in range(1,11):
   print(9,"*",i,"=",9*i)

# 5: WAP to print the table of n and n is given by user.   
t=int(input("enter a number to print its table"))
for i  in range(1,11):
    print(t,"*",i,"=",t*i)


# 6: WAP to add all the numbers from 1 to n and n is given by user.    
n=int(input("enter the number to calculate their sum"))
sum=0
for i in range(1,n+1):
    sum=sum+i
print(sum) 


# 7: WAP to find max amoung three numbers and input from user. [Try max() function]
a=10
b=5
c=15
print(max(a,b,c))


# 8: WAP to add all numbers divisible by 7 and 9 from 1 to n and n is given by the user.
n=int(input("enter number to add all numbers divisible by 7 and 9 from 1 to n"))
sum=0
for i in range(63,n+1):
    if (i%63==0):
        sum=sum+i
print(sum)   


# 9: WAP to add all prime numbers from 1 to n and n is given by the user.
n=int(input("enter number to add all prime numbers from 1 to n"))
sum=0
for i in range(2, n+1):
    flag=0
    for j in range(2,i):
        if (i%j==0):
            flag=1
            break
    if (flag==0):
        sum=sum+i    
print(sum)


# 10: WAP using function that add all odd numbers from 1 to n, n is given by the user.
def add(n):
    sum=0
    for i in range (1,n+1):
        if (i%2==1):
            sum=sum+i
    return sum
a=int(input("enter number to add all odd numbers from 1 to n"))
print(add(a))


# 11: WAP using function that add all prime numbers from 1 to n, n given by the user.
def prime(n):
    sum=0
    for i in range(2, n+1):
        flag=0
        for j in range(2,i):
            if (i%j==0):
                flag=1
                break
        if (flag==0):
            sum=sum+i    
    return (sum)
n=int(input("enter number to add all prime numbers from 1 to n"))
print(prime(n))