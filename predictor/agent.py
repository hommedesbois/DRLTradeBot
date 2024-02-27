import random
import numpy as np
import pandas as pd
from collections import deque
import tensorflow as tf
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.optimizer.set_jit(True)
from tensorflow import keras
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, BatchNormalization
from keras.models import Sequential, clone_model
import keras.backend as K
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


class DQLAgent:
    def __init__(self, hidden_units, learning_rate, batch,
                 train_env, test_env):
        self.train_env = train_env
        self.test_env = test_env
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.98
        self.learning_rate = learning_rate
        self.gamma = 0.5
        self.tau = 0.001
        self.batch = batch
        self.batch_size = 256
        self.mini_batch_size = 256
        self.max_treward = 0
        self.trewards = list()
        self.losses = list()
        self.sharpes = list()
        self.averages = list()
        self.performances = list()
        self.aperformances = list()
        self.vperformances = list()
        self.data = pd.DataFrame()
        self.memory = deque(maxlen=10000) 
       
        self.value_model = self._build_lstm_model(hidden_units, learning_rate)
        self.target_model = clone_model(self.value_model)
        self.target_model.set_weights(self.value_model.get_weights())


    def _build_lstm_model(self, hu, lr):
        model = Sequential()
        #model.add(Conv1D(filters=hu+2, kernel_size=3, activation='relu', input_shape=(self.train_env.lags, 
        #    self.train_env.n_features)))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(int(hu), return_sequences=True, input_shape=(self.train_env.lags, 
            self.train_env.n_features)))
        #model.add(BatchNormalization())
        model.add(LSTM(int(hu/2), return_sequences=False))
        #model.add(BatchNormalization())
        model.add(Dense(int(hu/2),activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dense(2, activation='linear'))
        model.compile(
            loss=self._huber_loss,
            optimizer=keras.optimizers.Adam(learning_rate=lr)
        )
        return model
    
    def _build_dense_model(self, hu, lr):
        model = Sequential()
        model.add(Dense(hu, activation ='relu', input_shape=(self.train_env.lags, self.train_env.features)))
        model.add(Dense(hu*2,activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=lr)
        )
        return model

    def _hard_update(self):
        self.target_model.set_weights(self.value_model.get_weights)

    def _soft_update(self):
        target_weights = self.target_model.get_weights()
        value_weights = self.value_model.get_weights()
        new_weights = [self.tau * tw + (1 - self.tau) * vw for tw, vw in zip(target_weights, value_weights)]
        self.target_model.set_weights(new_weights)

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        """Huber loss - Custom Loss Function for Q Learning"""
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def act(self, state):
        if random.random() <= self.epsilon:
            return self.train_env.action_space.sample()
        action = self.value_model.predict(state, verbose=0)[0]
        return np.argmax(action)
    
    def replay_batch(self, epsiode):
        # soft update for target network 
        self._soft_update()

        batch = random.sample(self.memory, self.batch_size)
        losses = []
        for i in range(0, self.batch_size, self.mini_batch_size):
            mini_batch = batch[i:i + self.mini_batch_size]
            states, actions, rewards, next_states, dones = map(np.array, zip(*mini_batch))
        
            #rewards = (rewards - rewards.mean()) / rewards.std()

            future_qs = self.target_model.predict(next_states, verbose=0)
            current_qs = self.value_model.predict(states, verbose=0)
        
            updated_qs = np.copy(current_qs)
            for i in range(self.mini_batch_size):
                if dones[i]:
                    updated_qs[i, actions[i]] = rewards[i]
                else:
                    updated_qs[i, actions[i]] = rewards[i] + self.gamma * np.amax(future_qs[i])

            loss = self.value_model.fit(states, updated_qs, epochs=1, verbose=False).history['loss'][0]
            losses.append(loss) 
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return np.mean(np.array(losses))
    
    def replay_serial(self, epsiode):
        losses = []
        # soft update for target network 
        self._soft_update() 

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            next_state = np.reshape(next_state, [1, self.train_env.lags,
                                       self.train_env.n_features])

            target = self.value_model.predict(state, verbose=0)
            future_q = self.target_model.predict(next_state, verbose=0)[0]
            #target = np.copy(current_q)
            if not done:
                reward += self.gamma * np.amax(future_q)
            target[0, action] = reward
           
            loss = self.value_model.fit(state, target, epochs=1, verbose=False).history["loss"][0]
            losses.append(loss)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return np.mean(np.array(losses))

    def learn(self, episodes):
        for e in range(1, episodes + 1):
            batch_state = self.train_env.reset()
            state = np.reshape(batch_state, [1, self.train_env.lags,
                                       self.train_env.n_features])
            treward = 0
            for _ in range(10000):
                action = self.act(state)
                next_batch_state, reward, done, info = self.train_env.step(action)
                # needed for predict in act function 
                next_state = np.reshape(next_batch_state,
                                [1, self.train_env.lags,
                                 self.train_env.n_features])
                self.memory.append([batch_state, action, reward, next_batch_state, done])
                state = next_state
                treward += reward
                if done:
                    #treward = _ + 1 # Reward is received for surviving 
                    self.trewards.append(treward)
                    av = sum(self.trewards[-5:]) / 5
                    perf = self.train_env.performance
                    sharpe = self.train_env.sharpe
                    self.sharpes.append(sharpe)
                    self.averages.append(av)
                    self.performances.append(perf)
                    self.aperformances.append(
                        sum(self.performances[-5:]) / 5)
                    self.max_treward = max(self.max_treward, treward)
                    
                    # I have put it here so that the loss has the same timestamp as the other paramters.
                    # This is possible due to the "break" statement
                    if len(self.memory) > self.batch_size:
                        if self.batch:
                            loss = self.replay_batch(e)
                        else:
                            loss = self.replay_serial(e)
                    
                        self.losses.append(loss)
                    else:
                        loss = 0.0
                    templ = 'episode: {:>3d}/{:<3d} | loss: {:>.5f} | '
                    templ += 'perf: {:>8.3%} | endurance: {:>4d} | total reward: {:>6.2f} | ' 
                    templ += 'epsilon: {:>3.2}'
                    print(templ.format(e, episodes, loss, (perf-1),
                                  (_ + 1), treward, self.epsilon))
                    break
            
            if e%25==0:
                modelname = 'models/{}_e{}_lags{}_tau{}_gamma_{}'
                self.value_model.save(modelname.format(self.train_env.symbol[0].split('_')[0], e, 
                                                       self.train_env.lags, str(self.tau).replace('.', '_'), 
                                                       str(self.gamma).replace('.', '_')))
            
            if e%555 == 0:
                self.validate(e, episodes)

        print()

    def validate(self, e, episodes):
        batch_state = self.test_env.reset()
        # add dimnesion to state 
        state = np.reshape(batch_state, [1, self.test_env.lags,
                                   self.test_env.n_features])
        treward = 0
        for _ in range(10000):
            action = np.argmax(self.value_model.predict(state, verbose=0)[0])
            next_batch_state, reward, done, info = self.test_env.step_val(action)
            # add dimnesion to state  
            state = np.reshape(next_batch_state, [1, self.test_env.lags,
                                   self.test_env.n_features])
            treward += reward 
            if done:
                perf = self.test_env.performance
                self.vperformances.append(perf)
                precision = (treward)  / (len(self.test_env.data) - self.test_env.lags)               
                if e % 50 == 0:
                    templ = 80 * '='
                    templ += '\nepisode: {:>3d}/{:<3d} | VALIDATION | precision: {:>.2%} |'
                    templ += 'perf: {:>7.3%} | eps: {:>.2f}\n'
                    templ += 80 * '='
                    print(templ.format(e, episodes, precision,
                                       (perf-1), self.epsilon))
                                    
                break