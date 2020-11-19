from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, LinearSVC

"""
Faster alternative for SVC with polynomial features. Uses mathematical technique called `kernel trick`.
"""

X, y = make_moons()

svm_clf_1 = Pipeline((
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),                   
    ('linear_svc', LinearSVC(C=0.1, loss="hinge"))    
))

svm_clf_2 = Pipeline((
    ('scaler', StandardScaler()),                   
    ('linear_svc', SVC(kernel="poly", degree=3, coef0=1, C=0.1))    # coef0 controls how high-degree polynomials influence classification versus low-degree polynomials.
))

svm_clf_1.fit(X, y)
svm_clf_2.fit(X, y)

acc_1 = accuracy_score(svm_clf_1.predict(X), y)
acc_2 = accuracy_score(svm_clf_2.predict(X), y)

print(acc_1, acc_2)
assert acc_1 <= acc_2
