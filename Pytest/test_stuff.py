data = {
    'Alicia': 100000,
    'Bob': 99817,
    'Carol': 122908,
    'Frank': 88123,
    'Eve': 93121,
    'Mark': 10201,
    'John': 99213,
    'Alex': 94515,
    'Christina': 993211
}



# Create a list of tuples, each containing an employee name and their salary,
# but only for employees with a salary larger or equal to 100,000
# Expected result:
#   [('Alicia', 100000), ('Carol', 122908), ('Christina', 993211)]
result = [(k, v) for k, v in data.items() if v >= 100000]