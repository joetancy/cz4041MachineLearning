import re
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from pandas.stats.api import ols

TourneySeeds = pd.read_csv('data/TourneySeeds.csv')
SampleSubmission = pd.read_csv('data/SampleSubmission.csv')
Seasons = pd.read_csv('data/Seasons.csv')
Teams = pd.read_csv('data/Teams.csv')
TourneySlots = pd.read_csv('data/TourneySlots.csv')
TourneyDetailedResults = pd.read_csv('data/TourneyDetailedResults.csv')
TourneyCompactResults = pd.read_csv('data/TourneyCompactResults.csv')

testTeams = SampleSubmission['Id'].str.split('_', expand=True)
testTeams = testTeams.as_matrix(columns=testTeams.columns[1:2])

winTeams = TourneyCompactResults.as_matrix(
    columns=TourneyCompactResults.columns[2:3])
loseTeams = TourneyCompactResults.as_matrix(
    columns=TourneyCompactResults.columns[4:5])
winScore = (TourneyCompactResults.as_matrix(
    columns=TourneyCompactResults.columns[3:4]))
loseScore = (TourneyCompactResults.as_matrix(
    columns=TourneyCompactResults.columns[5:6]))

#result = np.ones(1983)
result = np.subtract(winScore, loseScore)

TourneySeeds['SeedNum'] = TourneySeeds['Seed'].apply(
    lambda x: re.sub("[A-Z+a-z]", "", x, flags=re.IGNORECASE))

print(TourneySeeds.tail(10))

print(result)
teams = np.concatenate((winTeams, loseTeams), axis=1)

# model = Sequential()
# model.add(Dense(1, init='uniform', input_shape=(teams.shape[1:])))
# model.add(Dense(1, activation=('linear')))
#
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(teams, result, nb_epoch=10, batch_size=5)
#
# score = model.evaluate(teams, result, batch_size=5)
# print(score)
#
# predictions = model.predict(teams)
# print(predictions)
#
# np.savetxt("foo.csv", predictions, delimiter=",")
