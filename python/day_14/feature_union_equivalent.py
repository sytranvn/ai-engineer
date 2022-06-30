vect = CountVectorizer()
tfidf = TfidfTransformer()
txt_len = TextLengthExtractor()
clf = RandomForestClassifier()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)

X_train_len = txt_len.fit_transform(X_train)
X_train_features = hstack([X_train_tfidf, X_train_len])
clf.fit(X_train_features, y_train)

# predict on test data
X_test_counts = vect.transform(X_test)
X_test_tfidf = tfidf.transform(X_test_counts)

X_test_len = txt_len.transform(X_test)
X_test_features = hstack([X_test_tfidf, X_test_len])
y_pred = clf.predict(X_test_features)
