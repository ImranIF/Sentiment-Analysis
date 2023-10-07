from sklearn.feature_extraction.text import TfidfVectorizer
from multipledispatch import dispatch

@dispatch(list, list)
def vectorizeData(trainX, testX):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(trainX)
    trainXVectors = vectorizer.transform(trainX)
    testXVectors = vectorizer.transform(testX)

    return trainXVectors, testXVectors, vectorizer

@dispatch(list)
def vectorizeData(trainX):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(trainX)
    trainXVectors = vectorizer.transform(trainX)
    return trainXVectors, vectorizer
