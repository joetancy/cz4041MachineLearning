import numpy as np
import pandas as pd
import re
import tensorflow as tf

SampleSubmission = pd.read_csv('data/SampleSubmission.csv')
TourneySeeds = pd.read_csv('data/TourneySeeds.csv')
# Seasons = pd.read_csv('data/Seasons.csv')
# Teams = pd.read_csv('data/Teams.csv')
# TourneySlots = pd.read_csv('data/TourneySlots.csv')
# TourneyDetailedResults = pd.read_csv('data/TourneyDetailedResults.csv')
TourneyCompactResults = pd.read_csv('data/TourneyCompactResults.csv')
TourneyCompactResults['Difference'] = TourneyCompactResults[
    'Wscore'] - TourneyCompactResults['Lscore']

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

MatchDifference = TourneyCompactResults.groupby(['Wteam', 'Lteam'])[
    'Difference'].mean()
print(MatchDifference)

game_to_predict = pd.merge(game_to_predict, TourneySeeds[['Season', 'Team', 'SeedNum']].rename(
    columns={'Season': 'season', 'Team': 'team1', 'SeedNum': 'TeamSeed1'}), how='left', on=['season', 'team1'])
game_to_predict = pd.merge(game_to_predict, TourneySeeds[['Season', 'Team', 'SeedNum']].rename(
    columns={'Season': 'season', 'Team': 'team2', 'SeedNum': 'TeamSeed2'}), how='left', on=['season', 'team2'])

game_to_predict = pd.merge(game_to_predict, MatchDifference[
                           ['Wteam', 'Lteam']].rename(columns={'Wteam': 'team1', 'Lteam': 'team2'}), how='left', on=['team1', 'team2'])
# print(game_to_predict)
game_to_predict.to_csv(path_or_buf='diff.csv')
