'''
Linear Regression
refreence:
https://www.youtube.com/watch?v=q6ksri0LeDE&list=PLMrJAkhIeNNRpsRhXTMt8uJdIGz9-X_1-&index=19
https://www.youtube.com/watch?v=vNjLugdaGvs&list=PLMrJAkhIeNNRpsRhXTMt8uJdIGz9-X_1-&index=21'''
import matplotlib.pyplot as plt
import numpy as np
# Exp.1
# plt.rcParams['figure.figsize'] = [8, 8]
# plt.rcParams.update({'font.size': 18})
#
# x = 3  # true slope
# a = np.arange(-1, 2, 0.25)
# a = a.reshape(-1, 1)
# b = x * a + np.random.randn(*a.shape)
#
# plt.plot(a, a*x, Color='g', LineWidth=2, label='True line')
# plt.plot(a, b, 'x', Color='r', MarkerSize=10, label='Noisy data')
# U, S, VT = np.linalg.svd(a, full_matrices=False)
# xtile = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
# plt.plot(a, xtile*a, '--', Color='b', LineWidth=4, label='Regression')
# plt.xlabel('a')
# plt.ylabel('b')
#
# plt.grid(linestyle='--')
# plt.legend()
# plt.show()

# Exp.2
import os
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams.update({'font.size': 18})

# load dataset
H = np.loadtxt(os.path.join('./', 'DATA', 'housing.data'))
b = H[:, -1]  # house value
A = H[:, :-1]  # other factors
# pad with ones for nonzero offset
A = np.pad(A, [(0, 0), (0, 1)], mode='constant', constant_values=1)

# solve Ax=b using SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)
x = VT.T * np.linalg.inv(np.diag(S)) @ U.T @ b

fig = plt.figure()
ax1 = fig.add_subplot(121)
plt.plot(b, Color='k', LineWidth=2, label='housing value')
plt.plot(A@x, '-o', Color='r', LlineWidth=1.5, MarkerSize=6)