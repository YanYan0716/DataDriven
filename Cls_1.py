"""
image compression
reference:https://www.youtube.com/watch?v=H7qMMudo3e8&list=PLMrJAkhIeNNRpsRhXTMt8uJdIGz9-X_1-&index=6
"""
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.figsize'] = [16, 8]

A = imread('cls_1.jpg')
X = np.mean(A, -1)  # convert RGB to grayscale

img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()

# using SVD
U, S, VT = np.linalg.svd(X, full_matrices=False)  # full_matrices=False: using economy SVD
S = np.diag(S)
j = 0
for r in (5, 10, 20):
    # construct approximate
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r=' + str(r))
    plt.show()

plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('singular values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('singular values: comulative sum')
plt.show()