import pandas as pd
def m_grad(actual,exp,m,c,h):
    a=0
    for i in range(len(actual)):
        a=a+((actual[i]-(m+h)*exp[i]-c))**2-(actual[i]-(m)*exp[i]-c)**2
    return a

def c_grad(actual,exp,m,c,h):
    b=0
    for i in range(len(actual)):
        b=b+(actual[i]-m*exp[i]-(c+h))**2-(actual[i]-(m)*exp[i]-c)**2
    return b

data=pd.read_csv('Salary_Data.csv')
actual =data['Salary']
exp=data['YearsExperience']
h=0.01
alpha=0.01
m=100
c=200
for i in range(15000):
    m=m-alpha*m_grad(actual,exp,m,c,h)
    c=c-alpha*c_grad(actual,exp,m,c,h)
print(m,c)
