scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_train)

parameters = {
    'kernel': ['linear', 'rbf'],
    'C':[1, 10]
}

svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(scaled_data, y_train)
