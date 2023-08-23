def mea(a,b,n):
    num=int(0)
    for i in range(n):
        num=num+abs(a[i]-b[i])
    return num/n

def mse(a,b,n):
    num=int(0)
    for i in range(n):
        num+=pow((a[i]-b[i]),2)
    return num/n

def mape(a,b,n):
    sum=int(0)
    for i in range(n):
        sum=sum+abs(a[i]-b[i])/a[i]
    return sum/n

n=int(input("enter total no. elements:"))
a= []
p= []
print("enter actual")
for i in range(n):
    ele=int(input())
    a.append(ele)
print("enter prediction")
for i in range(n):
    ele=int(input())
    p.append(ele) 
print("mean absolute error:"+mea(a,p,n))
print("mean square error:"+mse(a,p,n))
print("mean absolute percentage error:"+mape(a,p,n))
