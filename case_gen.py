import sys
import random

if len(sys.argv) == 1:
	print('Missing n_cities')
	exit(0)

n_cities = int(sys.argv[1])

print(1)
print(n_cities)
for i in range(n_cities):
	print(f'{random.randint(0, 1000)} {random.randint(0, 1000)}')