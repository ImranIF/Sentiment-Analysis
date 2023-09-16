from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def trainModels(trainXVectors, trainY):
    classifierSVM = svm.SVC(kernel='linear')
    classifierSVM.fit(trainXVectors, trainY)

    classifierDecision = DecisionTreeClassifier()
    classifierDecision.fit(trainXVectors, trainY)

    classifierGaussianNaiveBayes = GaussianNB()
    classifierGaussianNaiveBayes.fit(trainXVectors.toarray(), trainY)

    classifierRandomForest = RandomForestClassifier(n_estimators=400, random_state=0)
    classifierRandomForest.fit(trainXVectors, trainY)

    return classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest