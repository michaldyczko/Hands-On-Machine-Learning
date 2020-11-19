import numpy as np
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR

"""
For linear data.
"""

SIZE = 1000

X = np.linspace(0, 1, num=SIZE)
y = X + np.random.normal(size=SIZE, scale=0.1)

X = X.reshape(-1, 1)

svm_reg_1 = LinearSVR(epsilon=0.5)

svm_reg_2 = LinearSVR(epsilon=0.01)

svm_reg_1.fit(X, y)
svm_reg_2.fit(X, y)

r2_1 = r2_score(y, svm_reg_1.predict(X))
r2_2 = r2_score(y, svm_reg_2.predict(X))

print(r2_1, r2_2)
assert r2_1 <= r2_2
