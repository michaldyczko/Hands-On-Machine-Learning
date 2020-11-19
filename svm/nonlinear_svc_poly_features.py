from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

"""
For non-linearly separable data. 
With complex datasets low degree can be unable to separate them, but high degree causes many features and slow training.
"""

X, y = make_moons()

svm_clf_1 = Pipeline((
    ('poly_features', PolynomialFeatures(degree=3)), # If X is not linearly separable, adding polynomial features can make it such
    ('scaler', StandardScaler()),                   
    ('linear_svc', LinearSVC(C=0.1, loss="hinge"))    
))

svm_clf_2 = Pipeline((
    ('poly_features', PolynomialFeatures(degree=3)), 
    ('scaler', StandardScaler()),                   
    ('linear_svc', LinearSVC(C=0.01, loss="hinge"))    
))

svm_clf_1.fit(X, y)
svm_clf_2.fit(X, y)

acc_1 = accuracy_score(svm_clf_1.predict(X), y)
acc_2 = accuracy_score(svm_clf_2.predict(X), y)

print(acc_1, acc_2)
assert acc_1 >= acc_2
