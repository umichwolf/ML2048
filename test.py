a = [[1,2],[3,4]]
with open('text.txt','w') as f:
    for line in a:
        f.write(str(line))
        f.write('\n')

b = []
with open('text.txt','r') as f:
    for line in f:
        b.append(eval(line))

print(b,type(b),type(b[0][0]))
