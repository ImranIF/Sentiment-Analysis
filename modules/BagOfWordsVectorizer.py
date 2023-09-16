from sklearn.feature_extraction.text import TfidfVectorizer
# def vectorizeData(trainX, testX):
def vectorizeData(trainX, testX):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(trainX)
    trainXVectors = vectorizer.transform(trainX)
    testXVectors = vectorizer.transform(testX)

    return trainXVectors, testXVectors, vectorizer
