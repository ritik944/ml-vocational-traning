def corner(a):
    temp=1
    for i in range(a):
        temp=temp*2
    return temp

def edges(c):
    return

 

d = input("enter a dimension:")
d=int(d)
print(corner(d))