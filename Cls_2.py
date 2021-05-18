'''
Unitary transformation and SVD
reference:
https://www.youtube.com/watch?v=MJAvyt9v0g4&list=PLMrJAkhIeNNRpsRhXTMt8uJdIGz9-X_1-&index=12&t=1s
https://github.com/dylewsky/Data_Driven_Science_Python_Demos/blob/master/CH01/CH01_SEC03_Rotation.ipynb
'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams.update({'font.size': 18})

theta = np.array([np.pi/15, -np.pi/9, -np.pi/20])
Sigma = np.diag([3, 1, 0.5])

# rotation about x axis
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(theta[0]), -np.sin(theta[0])],
    [0, np.sin(theta[0]), np.cos(theta[0])]
])
# rotation about y axis
Ry = np.array([
    [np.cos(theta[1]), 0, np.sin(theta[1])],
    [0, 1, 0],
    [-np.sin(theta[1]), 0, np.cos(theta[1])]
])
# rotation about z axis
Rz = np.array([
    [np.cos(theta[2]), -np.sin(theta[2]), 0],
    [np.sin(theta[2]), np.cos(theta[2]), 0],
    [0, 0, 1],
])

X = Rz @ Ry @ Rx @ Sigma
# use SVD
U, S, VT = np.linalg.svd(X, full_matrices=False)
X = U @ np.diag(S)

# plot sphere
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
u = np.linspace(-np.pi, np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# plot the surface
surf1 = ax1.plot_surface(x, y, z, cmap='jet', alpha=0.6, facecolors=plt.cm.jet(z), linewidth=0.5, rcount=30, ccount=30)
surf1.set_edgecolor('k')
ax1.set_xlim3d(-2, 2)
ax1.set_ylim3d(-2, 2)
ax1.set_zlim3d(-2, 2)

xR = np.zeros_like(x)
yR = np.zeros_like(y)
zR = np.zeros_like(z)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        vec = [x[i, j], y[i, j], z[i, j]]
        vecR = X @ vec
        xR[i, j] = vecR[0]
        yR[i, j] = vecR[1]
        zR[i, j] = vecR[2]
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(xR, yR, zR, cmap='jet', alpha=0.6, facecolors=plt.cm.jet(z), linewidth=0.5, rcount=30, ccount=30)
surf2.set_edgecolor('k')
ax2.set_xlim3d(-2, 2)
ax2.set_ylim3d(-2, 2)
ax2.set_zlim3d(-2, 2)
plt.show()