parameters = {
    'kernel': ['linear', 'rbf'],
    'C':[1, 10]
}

svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
