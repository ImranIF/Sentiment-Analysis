import csv
import json

csvFile = '../dataset/IMDB-Dataset.csv'
jsonFile = '../dataset/IMDB-Dataset.json'

dataList = []

with open(csvFile, 'r') as csvData:
    csvReader = csv.DictReader(csvData)
    for row in csvReader:
        dataList.append(row)

with open(jsonFile, 'w') as jsonData:
    for data in dataList:
        jsonData.write(json.dumps(data)+'\n')
    # json.dump(dataList, jsonData)