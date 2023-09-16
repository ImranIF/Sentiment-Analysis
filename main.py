import streamlit as st
# import threading
import modules
import time
import matplotlib.pyplot as plt
from textblob import TextBlob

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon='assets/If-Logo2.jpeg',
)
page = st.sidebar.selectbox(label='Controller', options=('Sentiment Analysis', 'Model Accuracy Comparison'))

with open('css/style.css') as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def saveSessionState(classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest,
                     testXVectors, testY, trainXVectors, vectorizer):
    st.session_state['classifierSVM'] = classifierSVM
    st.session_state['classifierDecision'] = classifierDecision
    st.session_state['classifierGaussianNaiveBayes'] = classifierGaussianNaiveBayes
    st.session_state['classifierRandomForest'] = classifierRandomForest
    st.session_state['testXVectors'] = testXVectors
    st.session_state['testY'] = testY
    st.session_state['trainXVectors'] = trainXVectors
    st.session_state['vectorizer'] = vectorizer


def loadSessionState():
    return st.session_state['classifierSVM'], st.session_state['classifierDecision'], st.session_state[
        'classifierGaussianNaiveBayes'], st.session_state['classifierRandomForest'], st.session_state['testXVectors'], \
        st.session_state['testY'], st.session_state['trainXVectors'], st.session_state['vectorizer']


file = 'dataset/IMDB-Dataset.json'
reviews = modules.loadReviews(file)
trainingStatus = st.empty()
if 'classifierSVM' not in st.session_state or 'classifierDecision' not in st.session_state or 'classifierGaussianNaiveBayes' not in st.session_state or 'classifierRandomForest' not in st.session_state:
    trainX, trainY, testX, testY = modules.prepareData(reviews)
    trainXVectors, testXVectors, vectorizer = modules.vectorizeData(trainX, testX)

    trainingStatus.write('Training models. Please wait a moment...')
    classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest = modules.trainModels(
        trainXVectors, trainY)
    trainingStatus.write('Models trained successfully!')
    saveSessionState(classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest,
                     testXVectors, testY, trainXVectors, vectorizer)

classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest, testXVectors, testY, trainXVectors, vectorizer = loadSessionState()

if classifierSVM and classifierDecision and classifierRandomForest and classifierGaussianNaiveBayes:
    modelNames = ['SVM', 'Decision Tree', 'Naive Bayes', 'Random Forest']
    classifierList = [classifierSVM, classifierRandomForest, classifierDecision, classifierGaussianNaiveBayes]
    accuracies = [modules.evaluateModel(classifier, testXVectors.toarray(), testY) for classifier in classifierList]
    trainingStatus.empty()

    if page == 'Sentiment Analysis':
        st.header('Sentiment Analysis')
        with st.form(key='nlpForm'):
            userInput = st.text_area('Enter text',
                                     placeholder='E.g: The earth temperature has been substantially reducing! It is quite shocking indeed!')

            if st.form_submit_button('Analyse Sentiment'):
                if userInput:
                    with st.spinner('Predicting sentiment...'):
                        time.sleep(1)
                        models = {
                            'SVM': classifierSVM,
                            'Random Forest': classifierRandomForest,
                            'Decision Tree': classifierDecision,
                            'Naive Bayes': classifierGaussianNaiveBayes
                        }
                        prediction = {}
                        for model in models:
                            prediction[model] = modules.analyzeSentiment([userInput], vectorizer, models[model])
                            print(prediction[model])
                        st.info('Results')
                        for key, values in prediction.items():
                            st.write('{}: {}'.format(key, values))
                else:
                    st.write('You entered no text!')
    elif page == 'Model Accuracy Comparison':
        st.subheader('Model Accuracy Comparison')
        labelColors = ['red', 'green', 'blue', 'purple']
        modules.pieChartGenerator(st, plt, accuracies, modelNames, labelColors)
        modules.barGenerator(st, plt, accuracies, modelNames, labelColors)

else:
    st.write('Error: Models were not trained successfully. Please check the training process.')
