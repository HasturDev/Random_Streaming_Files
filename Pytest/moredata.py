data2 = [
    'lambda functions are anonymous functions.',
    'anonymous functions dont have a name.',
    'functions are objects in Python.',
]

# Create a list of tuples, each consisting of a boolean value and the original
# string. The Boolean value indicates whether or not the string 'anonymous'
# appears in the original string.
# Expected result:
#   [(True, 'lambda functions are anonymous functions.'),
#    (True, 'anonymous functions dont have a name.'),
#    (False, 'functions are objects in Python.')]
print(list(map(lambda x: (True, x) if 'anonymous' in x else (False, x), data2)))
