import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

"""
For linearly separable data.
"""

iris = datasets.load_iris()
X = iris["data"][:, (2,3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica=1, others=0

# To prevent overfitting set smaller C parameter

svm_clf = Pipeline((
    ('scaler', StandardScaler()),                   # SVM is sensitive to scale
    ('linear_svc', LinearSVC(C=1, loss="hinge"))    # LinearSVC is much faster than SVC(kernel="linear", C=1), SGDClassifier(loss="hinge", alpha=1/(m*C)) converges slow
))

svm_clf.fit(X, y)

assert svm_clf.predict([[5.5, 1.7]]) == np.array([1.])
