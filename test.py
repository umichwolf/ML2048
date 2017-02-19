list1 = [1,2,2,3,6]

list2 = [3,1,2,4,5]

print list1
print list2

list1[:] = list2[:]

print list1
print list2

list2[0] = 8

print list1
print list2

