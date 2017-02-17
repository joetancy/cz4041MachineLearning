import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from pandas.stats.api import ols

rng = np.random

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
set2 = compact_results[['LSeedNum', 'WSeedNum']].rename(
    columns={'LSeedNum': 'Team1Seed', 'WSeedNum': 'Team2Seed'})
set2['Team1Win'] = 0
full_set = pd.concat([set1, set2], ignore_index=True)
full_set['Team1Seed'] = pd.to_numeric(full_set['Team1Seed'])
full_set['Team2Seed'] = pd.to_numeric(full_set['Team2Seed'])
full_set['Team1Win'] = pd.to_numeric(full_set['Team1Win'])

train_X = full_set['Team2Seed'] - full_set['Team1Seed']
train_Y = full_set['Team1Win']

winTeams = TourneyCompactResults.as_matrix(
    columns=TourneyCompactResults.columns[2:3])
loseTeams = TourneyCompactResults.as_matrix(
    columns=TourneyCompactResults.columns[4:5])

predict = game_to_predict['TeamSeed2'] - game_to_predict['TeamSeed1']

learning_rate = 0.01
training_epochs = 10
display_step = 5
n_samples = train_X.shape[0]
# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

pred = tf.add(tf.multiply(X, W), b)
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=",
          sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
