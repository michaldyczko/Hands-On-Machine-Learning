from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, LinearSVC

"""
Adding similarity features is a technique to tackle nonlinear problems. Similarity feature is computed using `Gaussian Radial Basis Function`.
We set one instance of X as landmark l, and then for x:

\phi(x, l) = exp(-\gamma||x-l||^2)

The simplest approach is to create as many features as possible landmarks equal to number of observations, but it creates very many features.
Again with the help of `kernel trick` one can achieve similar results but without computing additional features.
"""

X, y = make_moons()

svm_clf_1 = Pipeline((
    ('scaler', StandardScaler()),                   
    ('linear_svc', SVC(kernel="rbf", gamma=5, C=0.1))    # The higher the gamma value, the less values of similarity features, thus narrower margin => leads to overfitting.
))

svm_clf_2 = Pipeline((
    ('scaler', StandardScaler()),                   
    ('linear_svc', SVC(kernel="rbf", gamma=0.01, C=0.1))    # The higher the gamma value, the less values of similarity features, thus narrower margin => leads to overfitting.
))

svm_clf_1.fit(X, y)
svm_clf_2.fit(X, y)

acc_1 = accuracy_score(svm_clf_1.predict(X), y)
acc_2 = accuracy_score(svm_clf_2.predict(X), y)

print(acc_1, acc_2)
assert acc_1 > acc_2
