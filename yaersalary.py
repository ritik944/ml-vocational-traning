import pandas as pd
def m(a,b):
    ai=0
    xi=0
    aixi=0
    xisqr=0
    n=0
    for i in range(len(a)):
            ai+=a[i]
            xi+=b[i]
            aixi+=a[i]*b[i]
            xisqr+=b[i]*b[i]
            n+=1
    return ((n*aixi)-(ai*xi))/(n*xisqr-(xi*xi))


def c(a,b):
    x=len(a)
    ai=int(0)
    xi=int(0)
    n=int(0)
    for i in range (len(a)):
        ai=ai+a[i]
        xi=xi+b[i]
        n=n+1
    return (ai-m(a,b)*xi)/n
        
        
        
df=pd.read_csv('salary_Data.csv')
ai=[]
bi=[]
for row in df.Salary:
    ai.append(row)
    # bi.append(df.Salary)
for row in df.YearsExperience:
    bi.append(row) 
# print(ai)
# print(bi)
print("y=",int(m(ai,bi)),"x+",int(c(ai,bi)))