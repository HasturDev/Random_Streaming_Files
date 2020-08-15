data = '''
We spent several years building our own database engine,
Amazon Aurora, a fully-managed MySQL and PostgreSQL-compatible
service with the same or better durability and availability as
the commercial engines, but at one-tenth of the cost. We were
not surprised when this worked.
'''

# Detect the first occurrence of the string 'SQL' in the text above, and
# return it plus the 18 characters before and 18 characters after it.
# Expected result:
#   'a fully-managed MySQL and PostgreSQL-co'
result = lambda x, q: x[x.find(q)-18:x.find(q)+18] if q in x else -1
print(result(data, 'SQL'))