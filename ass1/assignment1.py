import numpy as np
import matplotlib.pyplot as plt

# part 1
lam = [0, 1, 2, 3]

r = np.linspace(-8, 8, num=100000)

for i in lam:
    loss = 0.5*(np.square(r-1) + np.square(r+1)) + i*np.abs(1-r)
    plt.plot(r, loss, label='{}'.format(i))

plt.legend(title='lambda')
plt.xlabel('r')
plt.ylabel('L')
plt.show()

# part 2
#r1 = np.linspace(-5, 5, 10)
#r2 = np.linspace(-5, 5, 10)
#R1, R2 = np.meshgrid(r1, r2)
#Z = np.abs(R1 - R2)
#plt.contour(R1, R2, Z)
#plt.show()
