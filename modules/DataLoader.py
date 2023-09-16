import json
from . import SentimentContainer
def loadReviews(file):
    reviews = []
    cnt = 0
    with open(file) as datasetFile:
        for eachLine in datasetFile:
            if cnt >= 5000:
                break
            review = json.loads(eachLine)
            # print(review)
            reviews.append(SentimentContainer.Review(review['review'], review['sentiment']))
            # print(len(reviews))
            cnt+=1
    return reviews

