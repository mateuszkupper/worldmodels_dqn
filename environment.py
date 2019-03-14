import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
from PIL import Image
import utils
import time
from env import make_env
from vae.vae import ConvVAE
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size

class CarRacing:

    # Parameters
    # - type: Name of environment. Default is classic Car Racing game, but can be changed to introduce perturbations in environment
    # - history_pick: Size of history
    # - seed: List of seeds to sample from during training. Default is none (random games)
    def __init__(self, type="CarRacing", history_pick=4, seed=None, detect_edges=False, detect_grass=False, flip=False):
        self.name = type + str(time.time())
        random.seed(30)
        self.env = make_env('CarRacing-v0', random.randint(1,10000000), render_mode = False, full_episode = True)
        self.image_dimension = [64,64]
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.image_dimension)
        self.action_space_size = 5
        self.state_shape = [None, self.history_pick] + list(self.image_dimension)
        self.history = []
        self.action_dict = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 0.8], 4: [0, 0, 0]}
        self.seed = seed
        self.detect_edges = detect_edges
        self.detect_grass = detect_grass
        self.flip = flip
        self.flip_episode = False
        self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)
        self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)
        self.vae.load_json('vae/vae.json')
        self.rnn.load_json('rnn/rnn.json')

    # returns a random action
    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        if self.flip_episode and action <= 1:
            action = 1 - action
        return self.action_dict[action]

    # resets the environment and returns the initial state
    def reset(self, test=False):
        self.state_rnn = rnn_init_state(self.rnn)
        if self.seed:
            self.env.seed(random.choice(self.seed))
        self.flip_episode = random.random() > 0.5 and not test and self.flip
        state, self.state_rnn = self.encode_obs(self.env.reset(), self.state_rnn, np.array([0.5, 0.2, 0.8]))
        return state, 1

    # take action 
    def step(self, action, test=False):
        action = self.map_action(action)
        total_reward = 0
        n = 1 if test else random.choice([2, 3, 4])
        for i in range(n):
            next_state, reward, done, info = self.env.step(action)
            next_state, self. state_rnn = self.encode_obs(next_state, self.state_rnn, action)
            total_reward += reward
            info = {'true_done': done}
            if done: break   
        return next_state, total_reward, done, info, 1

    def render(self):
        self.env.render()

    # process state and return the current history
    def process(self, state):
        self.add_history(state)
        in_grass = utils.in_grass(state)
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.image_dimension)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1, 1))
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result, in_grass

    def add_history(self, state):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        #temp = utils.process_image(state, detect_edges=self.detect_edges, flip=self.flip_episode)
        self.history.append(state)

    def __str__(self):
    	return self.name + '\nseed: {0}\nactions: {1}'.format(self.seed, self.action_dict)

    def encode_obs(self, obs, prev_state, action):
        # convert raw obs to z, mu, logvar
        result = np.copy(obs).astype(np.float)/255.0
        result = result.reshape(1, 64, 64, 3)
        mu, logvar = self.vae.encode_mu_logvar(result)
        mu = mu[0]
        logvar = logvar[0]
        s = logvar.shape
        z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
        h = rnn_output(prev_state, z, 4)
        next_state = rnn_next_state(self.rnn, z, np.array(action), prev_state)
        return np.concatenate([h, z]), next_state