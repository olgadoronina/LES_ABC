import numpy as np
import math as m
import matplotlib.pyplot as plt

v_s = 0.1
n = 3
r = np.power((v_s*n*m.gamma(n/2)/ 2/np.power(np.pi, n/2)), 1/n)



def v_sphere(rad, n):
    return 2*np.power(np.pi, n/2)*rad**n/(n*m.gamma(n/2))

cube =[]
sp =[]
n =[]
for i in range(3, 11):
    n.append(i)
    sp.append(v_sphere(r, i))
print(sp)
plt.scatter(n, sp, color='b')
plt.xlabel('dimension')
plt.ylabel('percent of accepted')
plt.show()