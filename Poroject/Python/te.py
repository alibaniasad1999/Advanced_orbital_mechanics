import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import models
from tensorflow.keras import backend as K
import random


# Define the equations of motion for the CRTBP environment 
class CRTBPEnv(gym.Env):
    def __init__(self, initial_condition, finial_state, mu):
        self.initial_condition = initial_condition
        self.finial_state = finial_state
        self.state = None
        if mu == None:
            self.mu = 0.012277471
        else:
            self.mu = mu
        # Define the action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-.1, high=.1, shape=(2,), dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=2, high=-2, shape=(4,), dtype=np.float32)

    def step(self, action):
        mu = self.mu
        # CRTBP equations in X-Y plane
        # X = x, y, vx, vy
        # U = mu
        # dot_X = f(X, U) = AX + BU
        # y = Cx + DU
        X = self.state.T
        A = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1-(1-mu)/((X[0]+mu)**2+X[1]**2)**(3/2)-mu/((X[0]-1+mu)**2+X[1]**2)**(3/2), 0, 0 , 2],
                        [0, 1-(1-mu)/((X[0]+mu)**2+X[1]**2)**(3/2)-mu/((X[0]-1+mu)**2+X[1]**2)**(3/2), -2, 0]])
        B = np.array([[0],
                     [0],
                     [mu*(1-mu)/((X[0]+mu)**2+X[1]**2)**(3/2)+mu*(1-mu)/((X[0]-1+mu)**2+X[1]**2)**(3/2)],
                        [0]])
        C = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        D = np.array([0, 0])
        if isinstance(action, tf.Tensor):
            action = action.numpy()[0]
        if len(action) == 1:
            action = action[0]
        dot_X = np.dot(A, X) + B.T + np.array([0, 0, action[0], action[1]])

        observation = X + dot_X

        # Calculate the reward
        reward = -np.linalg.norm(observation - self.finial_state)

        # Check if the episode is done
        done = bool(np.linalg.norm(observation - self.finial_state) < 0.001)

        info = {}
        return observation, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = self.initial_condition
        return self.initial_condition

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass


# create earth moon three body problem #
# initial condition #
initial_condition = np.array([0.994, 0, 0, -2.00158510637908252240537862224])
# finial state #
finial_state = np.array([0.994, 0, 0, -2.00158510637908252240537862224])
# create the environment #
env = CRTBPEnv(initial_condition, finial_state, mu = 0.012277471)
# reset the environment #
env.reset()
# take a random action #
action = env.action_space.sample()
# take a step #
observation, reward, done, info = env.step(action)
# print the observation #
print(observation)
# print the reward #
print(reward)
# print the done #
print(done)
# print the info #
print(info)
# print the action #
print(action)


# denife Q-learning NN and agent #


# define the replay buffer #
class ReplayBuffer(object):
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        actions = [i.numpy()[0][0] for i in actions]
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# define the Q-learning NN #
class Q_NN(keras.Model):
    def __init__(self):
        super(Q_NN, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(2, activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# define the agent for continuous action space #
class Agent(object):
    def __init__(self, env, buffer_capacity=10000, batch_size=32, gamma=0.99, lr=0.001):
        self.env = env
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.q_nn = Q_NN()
        self.q_nn_target = Q_NN()
        self.optimizer = optimizers.Adam(self.lr)
        self.loss = losses.MeanSquaredError()
        self.metric = metrics.MeanSquaredError()

    def update_target(self):
        self.q_nn_target.set_weights(self.q_nn.get_weights())

    def act(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)
        action = self.q_nn(state)
        return action

    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = [tf.convert_to_tensor(i, dtype=tf.float32) for i in states]
        actions = [tf.convert_to_tensor(i, dtype=tf.float32) for i in actions]
        rewards = [tf.convert_to_tensor(i, dtype=tf.float32) for i in rewards]
        next_states = [tf.convert_to_tensor(i, dtype=tf.float32) for i in next_states]
        dones = [tf.convert_to_tensor(i, dtype=tf.float32) for i in dones]

        with tf.GradientTape() as tape:
            y_true = rewards + self.gamma * [tf.reduce_max(self.q_nn_target(i) , axis=1) for i in next_states] * ([i-1 for i in dones])
            y_pred = tf.reduce_sum(self.q_nn(states) * actions, axis=1)
            loss = self.loss(y_true, y_pred)
            metric = self.metric(y_true, y_pred)
        grads = tape.gradient(loss, self.q_nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_nn.trainable_variables))
        return loss, metric

    def save(self, path):
        self.q_nn.save_weights(path)

    def load(self, path):
        self.q_nn.load_weights(path)



# define the train function #
def train(agent, env, replay_buffer, episodes=1000, batch_size=32):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            episode_reward += reward
            if len(replay_buffer.buffer) > batch_size:
                loss, metric = agent.train()
            if done:
                break
            state = next_state
        rewards.append(episode_reward)
        if episode % 10 == 0:
            print('Episode: {}, Reward: {}, Loss: {}, Metric: {}'.format(episode, np.mean(rewards[-10:]), loss, metric))
        if episode % 100 == 0:
            agent.update_target()


# create the replay buffer #
# replay_buffer = ReplayBuffer()
# create the agent #
agent = Agent(env)
# train the agent #
train(agent, env, agent.buffer, episodes=1000, batch_size=32)
# save the model #
agent.q_nn.save('Q_NN.h5')
# load the model #
agent.q_nn = models.load_model('Q_NN.h5')
# reset the environment #
env.reset()
# take a random action #
action = env.action_space.sample()
# take a step #
observation, reward, done, info = env.step(action)
# print the observation #
print(observation)
# print the reward #
print(reward)
# print the done #
print(done)
# print the info #
print(info)
# print the action #
print(action)
# print the Q-value #
print(agent.q_nn.predict(np.reshape(observation, [1, 4])))
# print the Q-value #
print(agent.q_nn.predict(np.reshape(observation, [1, 4])))
# print the Q-value #
