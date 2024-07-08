import numpy as np
import random
from os import path
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv2DTranspose, BatchNormalization, UpSampling2D, Reshape
from keras.models import load_model
import wandb


class agent:
    def __init__(self,discount=0.95,exploration_rate=0.9,decay_factor=0.9999, learning_rate=0.1):
        self.discount = discount # How much we appreciate future reward over current
        self.exploration_rate = exploration_rate # Initial exploration rate
        self.decay_factor = decay_factor
        self.learning_rate = learning_rate

        wandb.config.update({'model_name':'3xConv(64,3x3,relu)-Pool(2x2),Dense(64,relu),Dense(32,relu),Dense(1,sigmoid)(adam,mse)'})
        if(path.exists(self._getModelFilename())):
            self.model = load_model(self._getModelFilename())
        else:
            self.model = Sequential()
            self.model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(544,725,4)))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(64, kernel_size=3, activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(64, kernel_size=3, activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(64,activation="relu"))
            self.model.add(Dense(32,activation="relu"))
            self.model.add(Dense(1,activation="sigmoid"))
            self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def _getModelFilename(self):
        return "geodash.h5"

    def saveModel(self):
        self.model.save(self._getModelFilename())
        wandb.save(self._getModelFilename())        

    def get_next_action(self, state):
        if random.random() < self.exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.random_action()
        else:
            return self.greedy_action(state)

    def greedy_action(self, state):
        return self.getQ(state) > 0.5
    def random_action(self):
        return random.random() > 0.5

    def getQ(self,state):
        state_to_predict = np.expand_dims(state,0)
        action_prediction = self.model.predict(state_to_predict)
        return action_prediction[0][0]

    def train(self, old_state, new_state, reward):
        
        old_state_prediction = self.getQ(old_state)
        new_state_prediction = self.getQ(new_state)

        old_state_prediction = ((1-self.learning_rate) * old_state_prediction) + (self.learning_rate * (reward + self.discount * new_state_prediction))

        x = np.expand_dims(old_state,0)
        y = np.expand_dims(old_state_prediction,0)
        self.model.fit(x,y,verbose=0)

    def update(self, old_state, new_state, reward):        
        self.train(old_state, new_state, reward)
        self.exploration_rate *= self.decay_factor
