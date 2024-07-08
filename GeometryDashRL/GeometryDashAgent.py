import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=self.state_shape))
        model.add(layers.Conv2D(64, (4, 4), strides=2, activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
        return model

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = (reward + 0.99 * np.amax(self.model.predict(next_state, verbose=0)[0]))
        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
