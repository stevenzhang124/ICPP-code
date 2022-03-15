import random
from random import expovariate

random.seed(0)
lmbda = 1
limit = 50
timestamps = []

timestamp = 0.0
while timestamp <= limit:
    timestamp += expovariate(lmbda)
    timestamps.append(round(timestamp, 2))
print(len(timestamps))
print(timestamps)


sum_tmp = 0
for i in range(len(timestamps)-5, len(timestamps)):
	if timestamps[i] < 50:
		sum_tmp = sum_tmp + 50 - timestamps[i]

print(sum_tmp/len(timestamps))

