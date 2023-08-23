def corner(a):
    temp=1
    for i in range(a):
        temp=temp*2
    return temp

def edges(c):
    if(c==1):
        return 1
    elif(c==2):
        return 4
    elif(c>2):
        return edges(c-1)*2+corner(c-1)
 
# def faces(b):
    
    
d = input("enter a dimension:")
d=int(d)
print(corner(d))
print(edges(d))