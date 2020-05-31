# Question 8.35 of geometry manual
import numpy as np
import matplotlib.pyplot as plt
import math 
from coeffs import *
#Triangle sides
a = 10
b = 5.38
c = 5.38

#Triangle vertices
A,B,C = tri_vert(a,b,c)
m = B-C
n = np.matmul(omat,m) 
N=np.vstack((m,n))
p = np.zeros(2)
p[0] = m@A 
p[1] = n@B
#Intersection
D=np.linalg.inv(N.T)@p

print(A,B,C,D)

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AD = line_gen(D,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')
plt.plot(D[0], D[1], 'o')
plt.text(D[0] * (1 + 0.03), D[1] * (1 - 0.1) , 'D')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('./figs/triangle/tri_alt.pdf')
#plt.savefig('./figs/triangle/tri_alt.eps')
#subprocess.run(shlex.split("termux-open ./figs/triangle/tri_alt.pdf"))
#else
plt.show()

def printAngle(A, B, C): 

# Square of lengths be a2, b2, c2 
    a2 = lengthSquare(B, C) 
    b2 = lengthSquare(A, C) 
    c2 = lengthSquare(A, B) 

    # length of sides be a, b, c 
    a = math.sqrt(a2); 
    b = math.sqrt(b2); 
    c = math.sqrt(c2); 

# From Cosine law 
    alpha = math.acos((b2 + c2 - a2) /(2 * b * c)); 
    betta = math.acos((a2 + c2 - b2) /(2 * a * c)); 
    gamma = math.acos((a2 + b2 - c2) /(2 * a * b)); 

# Converting to degree 
    alpha = alpha * 180 / math.pi; 
    betta = betta * 180 / math.pi; 
    gamma = gamma * 180 / math.pi; 

# printing all the angles 
    print("alpha : %f" %(alpha)) 
    print("betta : %f" %(betta)) 
    print("gamma : %f" %(gamma)) 
# for traingle ABC
A = (5, 1.98605136) 
B = (0, 0) 
C = (10, 0) 

printAngle(A, B, C); 

# For traingle ABD
A = (5, 1.98605136) 
B = (0, 0) 
C = (5, 0) 

printAngle(A, B, C); 





