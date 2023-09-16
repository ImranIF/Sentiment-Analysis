from sklearn.metrics import f1_score
from . import SentimentContainer


def evaluateModel(model, testXVectors, testY):
    print(model)
    return model.score(testXVectors, testY)


def calculateF1Score(testY, predictions):
    return f1_score(testY, predictions, average=None,
                    labels=[SentimentContainer.Sentiment.POSITIVE, SentimentContainer.Sentiment.NEGATIVE], zero_division=1)
