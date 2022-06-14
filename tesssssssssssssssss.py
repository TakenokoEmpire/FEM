user_input = input("Enter string for variable name: \n")
globals()[user_input] = 50
print(apple)
print(type(apple))