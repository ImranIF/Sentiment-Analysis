import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
# import threading
import modules
import pickle
import time
import matplotlib.pyplot as plt
from textblob import TextBlob

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon='assets/If-Logo2.jpeg',
)
st.sidebar.image('assets/IF-Logo3.png', caption='Natural Language Processing')
page = st.sidebar.selectbox(label='Controller',
                            options=('Sentiment Analysis', 'Sentiment Feedback', 'Model Accuracy Comparison'))

css = """
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Dancing+Script:wght@400;500;600&family=Eczar:wght@400;500&display=swap');

body{
    color: rgb(250, 250, 250);
    background-color: rgb(14, 17, 23);
}

header.css-18ni7ap.ezrtsby2{
    color: rgb(250, 250, 250);
    background-color: rgb(14, 17, 23);
}

div.css-6qob1r.eczjsme3{
    background-color: #001C30;
}

div.st-be.st-bf.st-by.st-bz.st-c0.st-b3.st-c1.st-c2.st-bg.st-c3.st-c4.st-c5.st-c6{
    background: rgb(14, 17, 23);
    color: rgb(250, 250, 250);
}

.css-fg4pbf{
    background: rgb(14, 17, 23);
    color: rgb(250, 250, 250);
}

div.st-be.st-bf.st-by.st-bz.st-c0.st-b3.st-c1.st-c2.st-bg.st-c3.st-c4.st-c5.st-c6:hover, div.row-widget.stSelectbox:hover{
    cursor: pointer;
}

label.css-ue6h4q.e1y5xkzn3{
    letter-spacing: 0.25rem;
    /*padding-bottom: 1rem;*/
    /*padding-left: 3rem;*/
    padding: 0% 2% 2% 2%;
    color: rgb(250, 250, 250);
}

div.css-1j6rxz7{
    display: flex;
    justify-content: center;
    align-items: center;
    /*text-align: center;*/
}

div.css-r421ms.e10yg2by1{
    border: 1px solid rgba(250, 250, 250, 0.2);
}

div.css-1b0udgb{
    color: rgba(250, 250, 250, 0.6);
}

div.css-1j6rxz7.e1eexb540 p{
    font-family: 'Dancing Script', cursive;
    font-size: x-large;
 }

section.main.css-uf99v8.ea3mdgi5{
   background-color: #537895;
    background-image: linear-gradient(315deg, #537895 0%, #09203f 54%);
}

#MainMenu, footer {
    visibility: hidden;
}

button.css-7ym5gk.ef3psqc6{
    background-color: #6F1E51;
    color: blanchedalmond;
    border: none;
    width: 100%;
    padding: 1.3rem;
    display: block;
    transition: 0.4s;
}

button.css-7ym5gk.ef3psqc6 p{
    font-family: 'Eczar', serif;
    font-size: larger;
}

button.css-7ym5gk.ef3psqc6:hover{
    letter-spacing: 0.35rem;
    background-color:#580f3d;
    cursor: pointer;
    padding: 1.5rem;
}

button.css-19rxjzo.ef3psqc11:active{
    outline: none;
    background-color: #3a0225;
}

span.css-10trblm.e1nzilvr1{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'Amiri', serif;
    font-size: xx-large;
    color: rgb(250, 250, 250);
}
"""

st.markdown(
    f"""
    <style>
    {css}
    </style>
    """, unsafe_allow_html=True
)

# STREAMLIT_STATIC_PATH = Path(st.__path__[0]) / 'static'
# CSS_PATH = (STREAMLIT_STATIC_PATH / "css")
# if not CSS_PATH.is_dir():
#     CSS_PATH.mkdir()
#
# css_file = CSS_PATH / "style.css"
# if not css_file.exists():
#     shutil.copy("css/style.css", css_file)
# st.markdown(
#     f"""
#    <link rel="stylesheet" href="css/style.css" type="text/css"/>
#     """,
#     unsafe_allow_html=True
# )

# with open('static/style.css') as f:
#     st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def saveModel(classifierSVM, classifierDecision,  classifierGaussianNaiveBayes, classifierRandomForest, vectorizer):
    with open('models/svm_model.pkl', 'wb') as file:
        pickle.dump(classifierSVM, file)
    with open('models/decision_tree_model.pkl', 'wb') as file:
        pickle.dump(classifierDecision, file)
    with open('models/gaussian_naive_bayes_model.pkl', 'wb') as file:
        pickle.dump(classifierGaussianNaiveBayes, file)
    with open('models/random_forest_model.pkl', 'wb') as file:
        pickle.dump(classifierRandomForest, file)
    with open('models/vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)

def loadModel():
    with open('models/svm_model.pkl', 'rb') as file:
        classifierSVM = pickle.load(file)
    with open('models/decision_tree_model.pkl', 'rb') as file:
        classifierDecision = pickle.load(file)
    with open('models/gaussian_naive_bayes_model.pkl', 'rb') as file:
        classifierGaussianNaiveBayes = pickle.load(file)
    with open('models/random_forest_model.pkl', 'rb') as file:
        classifierRandomForest = pickle.load(file)
    with open('models/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest

def saveSessionState(classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest,
                     testXVectors, trainX, trainY, testX, testY, trainXVectors, vectorizer):
    st.session_state['classifierSVM'] = classifierSVM
    st.session_state['classifierDecision'] = classifierDecision
    st.session_state['classifierGaussianNaiveBayes'] = classifierGaussianNaiveBayes
    st.session_state['classifierRandomForest'] = classifierRandomForest
    st.session_state['testXVectors'] = testXVectors
    st.session_state['trainX'] = trainX
    st.session_state['trainY'] = trainY
    st.session_state['testX'] = testX
    st.session_state['testY'] = testY
    st.session_state['trainXVectors'] = trainXVectors
    st.session_state['vectorizer'] = vectorizer


def loadSessionState():
    # print("Loaded vectorizer shape:", st.session_state['vectorizer'].get_feature_names_out().shape)

    return st.session_state['classifierSVM'], st.session_state['classifierDecision'], st.session_state[
        'classifierGaussianNaiveBayes'], st.session_state['classifierRandomForest'], st.session_state['testXVectors'], \
        st.session_state['trainX'], st.session_state['trainY'], st.session_state['testX'], st.session_state['testY'], st.session_state['trainXVectors'], st.session_state['vectorizer']


file = 'dataset/IMDB-Dataset.json'
reviews = modules.loadReviews(file)
trainingStatus = st.empty()

if 'classifierSVM' not in st.session_state or 'classifierDecision' not in st.session_state or 'classifierGaussianNaiveBayes' not in st.session_state or 'classifierRandomForest' not in st.session_state:
    with st.spinner('Training models...'):

        trainX, trainY, testX, testY = modules.prepareData(reviews)
        trainXVectors, testXVectors, vectorizer = modules.vectorizeData(trainX, testX)

        classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest = loadModel()
        # classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest = modules.trainModels(
        #     trainXVectors, trainY)
        saveSessionState(classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest,
                         testXVectors, trainX, trainY, testX, testY, trainXVectors, vectorizer)
    # saveModel(classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest, vectorizer)
    trainingStatus.write("Models trained successfully! Please wait a moment")

classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest, testXVectors, trainX, trainY, testX, testY, trainXVectors, vectorizer = loadSessionState()

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
                        st.divider()
                        st.info('Results')
                        for key, values in prediction.items():
                            st.write('{}: {}'.format(key, values))
                else:
                    st.write('You entered no text!')
    elif page == 'Sentiment Feedback':
        st.subheader('Sentiment Feedback')
        with st.form(key='feedbackForm'):
            inputFeedback = st.text_area('Enter feedback text', placeholder='Kais is a suspicious character!')
            inputSentiment = st.radio('Actual Sentiment', ['Positive', 'Negative'])
            if st.form_submit_button('Submit Feedback'):
                if inputFeedback and inputSentiment:
                    with st.spinner('Analysing new feedback, Please wait till model is trained...'):
                        newTrainX = trainX + [inputFeedback]
                        newTrainY = trainY + [inputSentiment]
                        newTestX = testX + [inputFeedback]
                        newTestY = testY + [inputSentiment]

                        newTrainXVectors, newTestXVectors, newVectorizer = modules.vectorizeData(newTrainX, newTestX)
                        classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest = modules.trainModels(newTrainXVectors, newTrainY)
                        saveModel(classifierSVM, classifierDecision, classifierGaussianNaiveBayes, classifierRandomForest,
                                  newVectorizer)

                        saveSessionState(classifierSVM, classifierDecision, classifierGaussianNaiveBayes,
                                         classifierRandomForest,
                                         newTestXVectors, newTrainX, newTrainY, newTestX, newTestY, newTrainXVectors, newVectorizer)
                    st.success('Feedback submitted and model retrained successfully.')
                else:
                    st.error('Submission has been unsuccessful! Please resubmit with appropriate inputs')
    elif page == 'Model Accuracy Comparison':
        st.subheader('Model Accuracy Comparison')
        labelColors = ['red', 'green', 'blue', 'purple']
        modules.pieChart(st, accuracies, modelNames)
        modules.barChart(st, accuracies, modelNames, labelColors)
        modules.lineChart(st, accuracies, modelNames, labelColors)

else:
    st.write('Error: Models were not trained successfully. Please check the training process.')
