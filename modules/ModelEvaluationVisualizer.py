import numpy as np
import pydeck as pdk
import pandas as pd
import plotly.graph_objects as go
# import plotly.express as px
def pieChart(st, accuracies, modelNames):
    figure = go.Figure(data=[go.Pie(labels=modelNames, values=accuracies)])
    figure.update_layout(title='Pie Chart Visualizer')
    st.plotly_chart(figure)

def barChart(st, accuracies, modelNames, labelColors):
    accuracies = [acc * 100 for acc in accuracies]
    figure = go.Figure(go.Bar(x=modelNames, y=accuracies, text=accuracies, textposition='auto', marker_color=labelColors))
    figure.update_layout(title='Bar Chart Visualizer', xaxis_title='Models', yaxis_title='Accuracy (%)')
    st.plotly_chart(figure)

def lineChart(st, accuracies, modelNames, labelColors):
    figure = go.Figure([go.Scatter(x=modelNames, y=accuracies, mode='lines+markers', marker_color=labelColors)])
    figure.update_layout(title='Line Chart Visualizer', xaxis_title='Models', yaxis_title='Accuracy (%)')
    st.plotly_chart(figure)

def pieChartGenerator(st, plt, accuracies, modelNames, labelColors):
    figure, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(accuracies, labels=modelNames, autopct='%1.1f%%', startangle=90,
                                      colors=labelColors)
    colorInd = 0
    for text in texts:
        text.set(color=labelColors[colorInd])
        colorInd += 1
    ax.axis('equal')
    ax.set_facecolor('none')
    ax.set_title('Pie-Chart Visualizer', color='cyan')
    figure.patch.set_alpha(0)
    st.pyplot(figure)

def barGenerator(st, plt, accuracies, modelNames, labelColors):
    figure, ax = plt.subplots()
    positions = np.arange(len(modelNames))
    for i in range(len(modelNames)):
        ax.bar(positions[i], accuracies[i],  color=labelColors[i])
        ax.text(positions[i], accuracies[i], '{}%'.format(positions[i]), ha='center', color=labelColors[i])

    ax.set_xticks(positions)

    for tick, color in zip(ax.get_xticklabels(), labelColors):
        tick.set_color(color)

    ax.set_xticklabels(modelNames)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy (%)')
    ax.set_facecolor('none')
    ax.set_title('Bar Chart Visualizer', color='#A0BFE0')
    figure.patch.set_alpha(0)
    st.pyplot(figure)


