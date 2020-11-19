import numpy as np
from sklearn.metrics import r2_score
from sklearn.svm import SVR

"""
For non-linear data.
"""

SIZE = 1000

X = np.linspace(0, 100, num=SIZE)
y = X**2 + np.random.normal(size=SIZE, scale=0.1)

X = X.reshape(-1, 1)

svm_reg_1 = SVR(kernel="poly", degree=2, C=0.1, epsilon=0.1)

svm_reg_2 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)

svm_reg_1.fit(X, y)
svm_reg_2.fit(X, y)

r2_1 = r2_score(y, svm_reg_1.predict(X))
r2_2 = r2_score(y, svm_reg_2.predict(X))

print(r2_1, r2_2)
assert r2_1 <= r2_2
