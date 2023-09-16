from sklearn.model_selection import train_test_split
from . import SentimentContainer

def prepareData(reviews):
    training, test = train_test_split(reviews, random_state=42)
    trainContainer = SentimentContainer.ReviewContainer(training)
    testContainer = SentimentContainer.ReviewContainer(test)

    trainContainer.evenlyDistribute()
    testContainer.evenlyDistribute()

    trainX = trainContainer.getReviewText()
    trainY = trainContainer.getSentiment()
    # return trainX, trainY

    testX = testContainer.getReviewText()
    testY = testContainer.getSentiment()

    return trainX, trainY, testX, testY

