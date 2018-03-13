import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import linear_model

data = pd.read_csv('wine.csv')
original_headers = list(data.columns.values)
print(original_headers)

y = data.quality
X = data.drop('quality', axis=1)
# print(X.describe)
X = preprocessing.scale(X)
print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

rf = RandomForestClassifier(n_estimators=40, warm_start=True)
kc = rf.fit(X_train, y_train)
print(kc)
predictions = rf.predict(X_test)
print(predictions)
s = y_test.values
print(y_test.values)
length = len(predictions)
print("Length of predictions".format(length))
print(len(predictions))
count = 0
for i in range(length):
    if predictions[i] == s[i]:
        count = count + 1
# print("Length of count")
# print(count)
accuracy1 = count / length
# Obtaining the confidence score
print("Initial accuracy in model1 is:\n")
print(accuracy1)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

print(pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'], margins=True))

from sklearn.metrics import classification_report

classifyrep = classification_report(y_test, predictions)
print(classification_report(y_test, predictions))

# k-fold Stratified Cross Validation
sratifiedkf = StratifiedKFold(n_splits=10, shuffle=True)
average = 0.0
value = 0
for train_index, test_index in sratifiedkf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf = RandomForestClassifier(n_estimators=80, warm_start=True)
    kc = rf.fit(X_train, y_train)
    print(kc)
    predictions = rf.predict(X_test)
    print(predictions)
    s = y_test.values
    print(y_test.values)
    length = len(predictions)
    print("Length of predictions".format(length))
    print(len(predictions))
    count = 0
    for i in range(length):
        if predictions[i] == s[i]:
            count = count + 1
            print("Length of count")
            print(count)
    accuracy = count / length
    value = value + (accuracy - accuracy1)
    print("Test data accuracy in fold :\n {:.2f}".format(accuracy))


average = average + accuracy
average = average / 10
print("Average cross-validation is: {:.2f}".format(average))
print("Improvised accuracy in model1 is: \n {:.2f}".format(abs(value / 10)))


#linear regression -feature selection
#important features

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X=sel.fit_transform(X)


reg = linear_model.LinearRegression(normalize=True)

print(reg.fit(X, y))

a = reg.predict(X)
print(y)
# mean square error
print(np.mean((a - y) ** 2))

from sklearn.feature_selection import RFE

rfe = RFE(reg, 8)
fit = rfe.fit(X, y)
print(fit)
print(("Num Features: %d") % fit.n_features_)
print(("Selected Features: %s") % fit.support_)
print(("Feature Ranking: %s") % fit.ranking_)
X = data.drop(['citric acid', 'total sulfur dioxide', 'sulphates', 'quality'], axis=1)
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

# k-fold Stratified Cross Validation
sratifiedkf = StratifiedKFold(n_splits=10, shuffle=True)
average = 0.0
value = 0
for train_index, test_index in sratifiedkf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf = RandomForestClassifier(n_estimators=80, warm_start=True)
    kc = rf.fit(X_train, y_train)
    print(kc)
    predictions = rf.predict(X_test)
    print(predictions)
    s = y_test.values
    print(y_test.values)
    length = len(predictions)
    print("Length of predictions".format(length))
    print(len(predictions))
    count = 0
    for i in range(length):
        if predictions[i] == s[i]:
            count = count + 1
            # print("Length of count")
            # print(count)
    accuracy = count / length
    value = value + (accuracy - accuracy1)
    print("Test data accuracy in fold :\n {:.2f}".format(accuracy))

#average score
average = average + accuracy
average = average / 10
print("Average cross-validation is: {:.2f}".format(average))
print("Improvised accuracy after linear regression is: \n {:.2f}".format(abs(value / 10)))

