def analyzeSentiment(testSet, vectorizer, classifier):
    testVectors = vectorizer.transform(testSet)
    return classifier.predict(testVectors.toarray())