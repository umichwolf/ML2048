#First compare the number at the corners. Pick out the max one.
#If there are 2 or more the same, we compare the subcorners. Pick out the max one and so on.
#If all the numbers are the same such as the case at the beginning, then we pick one randomly.
#Once we fix the corner, we check the entries in the row and colume. We always choose our operation from the ones which won't affect the position of our corner.
#We always want to avoid the case that we have to move the fixed corner in the next step.
#We apply the operation that can give us larger entries.
#We apply the operation that eliminates entries as many as we can.
#Once the fixed corner has to be moved, we always check the possibility to move it back.
