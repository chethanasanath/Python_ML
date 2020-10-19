#Lambda expressions review. Generally Lambda expressions are used in conjunction with reduced functions like, map, filter..
#Below let us see how lambda expressions are written and for illustration, let us capture the value of lambda expression in a
#variable and print it.

#def square(x):
#    return x*x

#1.Get square of a num using Lambda expression
result = lambda x : x*x

print(result(7))

#2.Check if a num is even or odd using lambda expression
"""def iseven(x):
    if(x%2==0):
        return True
    else:
        return False"""

even = lambda x: x%2 ==0
print(even(6))

#3.Return first char of a string
first = lambda s:s[0]
print("First char of chethana: "+first("chethana"))

#4.Reverse a string
reversedStr = lambda s:s[::-1]
print("Reversed string of sanath is "+reversedStr("sanath"))

#5.Find sum of 2 numbers
adder = lambda x,y: x+y
print(adder(5,4))