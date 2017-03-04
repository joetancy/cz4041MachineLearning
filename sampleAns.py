import numpy as np
import pandas as pd
import re
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import matplotlib.pyplot as plt

TourneySeeds = pd.read_csv('data/TourneySeeds.csv')
SampleSubmission = pd.read_csv('data/SampleSubmission.csv')
Seasons = pd.read_csv('data/Seasons.csv')
Teams = pd.read_csv('data/Teams.csv')
TourneySlots = pd.read_csv('data/TourneySlots.csv')
TourneyDetailedResults = pd.read_csv('data/TourneyDetailedResults.csv')
TourneyCompactResults = pd.read_csv('data/TourneyCompactResults.csv')
TourneySeeds['SeedNum'] = TourneySeeds['Seed'].apply(
    lambda x: re.sub("[A-Z+a-z]", "", x, flags=re.IGNORECASE))

game_to_predict = pd.concat([SampleSubmission['Id'], SampleSubmission[
                            'Id'].str.split('_', expand=True)], axis=1)
game_to_predict.rename(
    columns={0: 'season', 1: 'team1', 2: 'team2'}, inplace=True)
game_to_predict['season'] = pd.to_numeric(game_to_predict['season'])
game_to_predict['team1'] = pd.to_numeric(game_to_predict['team1'])
game_to_predict['team2'] = pd.to_numeric(game_to_predict['team2'])
TourneySeeds['Season'] = pd.to_numeric(TourneySeeds['Season'])
TourneySeeds['Team'] = pd.to_numeric(TourneySeeds['Team'])
TourneySeeds['SeedNum'] = pd.to_numeric(TourneySeeds['SeedNum'])
game_to_predict = pd.merge(game_to_predict, TourneySeeds[['Season', 'Team', 'SeedNum']].rename(
    columns={'Season': 'season', 'Team': 'team1', 'SeedNum': 'TeamSeed1'}), how='left', on=['season', 'team1'])
game_to_predict = pd.merge(game_to_predict, TourneySeeds[['Season', 'Team', 'SeedNum']].rename(
    columns={'Season': 'season', 'Team': 'team2', 'SeedNum': 'TeamSeed2'}), how='left', on=['season', 'team2'])

compact_results = pd.merge(TourneyCompactResults, TourneySeeds[['Season', 'Team', 'SeedNum']].rename(
    columns={'Team': 'Wteam', 'SeedNum': 'WSeedNum'}), how='left', on=['Season', 'Wteam'])
compact_results = pd.merge(compact_results, TourneySeeds[['Season', 'Team', 'SeedNum']].rename(
    columns={'Team': 'Lteam', 'SeedNum': 'LSeedNum'}), how='left', on=['Season', 'Lteam'])

set1 = compact_results[['WSeedNum', 'LSeedNum']].rename(
    columns={'WSeedNum': 'Team1Seed', 'LSeedNum': 'Team2Seed'})
set1['Team1Win'] = 1

# full_set['Team1Seed'] = pd.to_numeric(full_set['Team1Seed'])
# full_set['Team2Seed'] = pd.to_numeric(full_set['Team2Seed'])
# full_set['Team1Win'] = pd.to_numeric(full_set['Team1Win'])

# print(full_set)


x = set1['Team2Seed'] - set1['Team1Seed']
y = set1['Team1Win']

# print(x.as_matrix())
# print(y.as_matrix())

predict = game_to_predict['TeamSeed2'] - game_to_predict['TeamSeed1']

# print(predict.as_matrix())

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Dense(1, activation=('sigmoid')))
model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(x, y, nb_epoch=10, batch_size=32)

predictions = model.predict(predict)
print(predictions)
plt.plot(set1[
    'Team2Seed'] - set1['Team1Seed'], set1['Team1Win'], 'ro', label='Original data')
plt.legend()
plt.show()
# np.savetxt("foo.csv", predictions, delimiter=",")
