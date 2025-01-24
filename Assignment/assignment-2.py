                                     #ASSIGNMENT-2
# 1. Create a List L that is defined as= [10, 20, 30, 40, 50, 60, 70, 80].
# i. WAP to add 200 and 300 to L.
# ii. WAP to remove 10 and 30 from L.
# iii. WAP to sort L in ascending order.
# iv. WAP to sort L in descending order.
L=[10, 20, 30, 40, 50, 60, 70, 80]
L.append(200)    
L.append(300)
L.remove(10)
L.remove(30)
print("original-",L)
L.sort()
print("ascending order-",L)
L.sort(reverse=True)
print("descending order-",L)


# 2. Create a tuple of marks scored as scores = (45, 89.5, 76, 45.4, 89, 92, 58, 45) and
# perform the following operations using tuple functions:
# i. Identify the highest score and its index in the tuple.
# ii. Find the lowest score and count how many times it appears.
# iii. Reverse the tuple and return it as a list.
# iv. Check if a specific score ‘76’ (input by the user) is present in the tuple and
# print its first occurrence index, or a message saying it’s not present.
T=(45, 89.5, 76, 45.4, 89, 92, 58, 45)
print(max(T))
print(T.index(max(T)))
print(min(T))
print(T.count(min(T)))
rev=list(T[::-1])
print(rev)
n=float(input("enter key"))
if (n in T):
    print(n," is present at ",T.index(n))
else :
    print(n, "is not present in tuple")


# 3. WAP to create a list of 100 random numbers between 100 and 900. Count and print
# the:
# i. All odd numbers
# ii. All even numbers
# iii. All prime numbers
import random
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True
random_numbers = [random.randint(100, 900) for i in range(100)]
print(random_numbers)
odd_count = 0
even_count = 0
prime_count = 0
odd_numbers = []
even_numbers = []
prime_numbers = []
for num in random_numbers:
    if num % 2 == 0:
        even_count += 1
        even_numbers.append(num)
    else:
        odd_count += 1
        odd_numbers.append(num)
    
    if is_prime(num):
        prime_count += 1
        prime_numbers.append(num)
print("Total odd numbers:",odd_count)
print("Total even numbers:" ,even_count)
print("Total prime numbers:" ,prime_count)


# 4. Consider the following two sets, A and B, representing scores of two teams in multiple
# matches. A = {34, 56, 78, 90} and B = {78, 45, 90, 23}
# WAP to perform the following opera􀆟ons using set functions:
# i. Find the unique scores achieved by both teams (union of sets).
# ii. Iden􀆟fy the scores that are common to both teams (intersection of sets).
# iii. Find the scores that are exclusive to each team (symmetric difference).
# iv. Check if the scores of team A are a subset of team B, and if team B's scores are
# a superset of team A.
# v. Remove a specific score 𝑋 (input by the user) from set A if it exists. If not, print
# a message saying it is not present.
A = {34, 56, 78, 90}
B = {78, 45, 90, 23}
union = A.union(B)
print("Union:",union)
intersection = A.intersection(B)
print("Intersection:" ,intersection)
symmetric_diff_set = A.symmetric_difference(B)
print("Symmetric Difference:", symmetric_diff_set)
is_subset = A.issubset(B)
is_superset = B.issuperset(A)
print("Is A a subset of B:",is_subset)
print("Is B a superset of A:" ,is_superset)
x = int(input("Enter a score to remove from team A: "))
if x in A:
    A.remove(x)
    print("Score ",x," removed from team A.")
    print(A) 
else:
    print("Score ",x," is not present in team A.")   


# 5.Write a program to rename a key city to a location in the following dictionary.
sample_dict = {
    "name": "Kelly",
    "age": 25,
    "salary": 8000,
    "city": "New York"
}
sample_dict["location"] = sample_dict.pop("city")
print(sample_dict)  