'''
using SVD to compute PCA
reference:https://www.youtube.com/watch?v=Oi4SJqJIL2E&list=PLMrJAkhIeNNRpsRhXTMt8uJdIGz9-X_1-&index=24
'''
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figures.figsize'] = [16, 8]


# make data
xC = np.array([2, 1])  # center of data
sig = np.array([2, 0.5])  # principal axes

theta = np.pi / 3
R = np.array([[np.cos(theta), np.sin(theta)],  # rotation matrix
             [np.sin(theta), np.cos(theta)]])

nPoints = 10000  # create 10000 points
X = R @ np.diag(sig) @ np.random.randn(2, nPoints) + \
    np.diag(xC) @ np.ones((2, nPoints))

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(X[0, :], X[1, :], '.', Color='k')
ax1.grad()
plt.xlim((-6, 8))
plt.ylim((-6, 8))

# compute PCA
Xavg = np.mean(X, axis=1)
B = X - np.tile(Xavg, (nPoints, 1)).T

# U:tell us about rotation in data,
# S:tell us about how stretched the data
U, S, VT = np.linalg.svd(B/np.sqrt(nPoints), full_matrices=0)
ax2 = fig.add_subplot(122)
ax2.plot(X[0,:], X[1,:], '.', Color='k')
ax2.grid()
plt.xlim((-6, 8))
plt.ylim((-6, 8))

theta = 2* np.pi* np.arange(0, 1, 0.01)
Xstd = U @ np.diag(S) @ np.array([np.cos(theta), np.sin(theta)])
