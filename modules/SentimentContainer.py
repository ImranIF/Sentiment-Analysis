import random


class Sentiment:
    POSITIVE = 'Positive'
    NEGATIVE = 'Negative'
    NEUTRAL = 'Neutral'


class Review:
    def __init__(self, review, sentiment):
        self.review = review
        self.sentiment = sentiment


class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def getReviewText(self):
        return [x.review for x in self.reviews]

    def getSentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenlyDistribute(self):
        negative = list(filter(lambda x: x.sentiment.lower() == Sentiment.NEGATIVE.lower(), self.reviews))
        positive = list(filter(lambda x: x.sentiment.lower() == Sentiment.POSITIVE.lower(), self.reviews))

        minimum = int(len(self.reviews)/2)
        shrinkPositive = positive[:minimum]
        shrinkNegative = negative[:minimum]
        self.reviews = shrinkNegative + shrinkPositive
        random.shuffle(self.reviews)
