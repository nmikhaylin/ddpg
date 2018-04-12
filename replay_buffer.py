import numpy as np
import tensorflow as tf
import collections

log = tf.logging
FLAGS = tf.flags.FLAGS

MAX_SIZE = 1000

class ReplayBuffer(object):
    def __init__(self):
        # Number of valid elements in the buffer.
        self._cur_size = 0
        self._i = 0
        self._old_states = None
        self._actions = None
        self._rewards = None
        self._dones = None
        self._new_states = None

    def printContents(self):
      log.info(self._old_states)
      log.info(self._actions)
      log.info(self._rewards)
      log.info(self._new_states)

    def sample(self, num_samples):
      samples = np.random.choice(self._cur_size, num_samples, replace=False)
      return (self._old_states[samples], self._actions[samples],
              self._rewards[samples], self._dones[samples],
              self._new_states[samples])

    def size(self):
      return self._cur_size

    def add(self, old_state, action, reward, done, new_state):
        # Not initialized
        if type(action) != np.ndarray:
          action = np.array([action])
        if type(reward) != np.ndarray:
          reward = np.array([reward])
        if type(done) != np.ndarray:
          done = np.array([done])
        if self._old_states is None:
            self._old_states = np.empty([MAX_SIZE] + list(old_state.shape), dtype=old_state.dtype)
            self._actions = np.empty([MAX_SIZE] + list(action.shape), dtype=action.dtype)
            self._rewards = np.empty([MAX_SIZE] + list(reward.shape), dtype=reward.dtype)
            self._dones = np.empty([MAX_SIZE] + list(done.shape), dtype=done.dtype)
            self._new_states = np.empty([MAX_SIZE] + list(new_state.shape), dtype=new_state.dtype)
        # Double the size of the array.
        if self._cur_size < MAX_SIZE:
          self._old_states[self._cur_size] = old_state
          self._actions[self._cur_size] = action
          self._rewards[self._cur_size] = reward
          self._dones[self._cur_size] = done
          self._new_states[self._cur_size] = new_state
          self._cur_size += 1
        # Start overwriting old values.
        else:
          self._old_states[self._i] = old_state
          self._actions[self._i] = action
          self._rewards[self._i] = reward
          self._dones[self._i] = done
          self._new_states[self._i] = new_state
          self._i = (self._i + 1) & MAX_SIZE
