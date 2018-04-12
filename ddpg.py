import collections
import math
import time

from keras import backend as K
from keras.layers import Dense,Input,Add,Concatenate
from keras.models import Model
from keras.optimizers import Adam
import gym
import numpy as np
import pandas as pd
import replay_buffer
import tensorflow as tf
import tensorflow.contrib.keras as keras

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean("load_model_from_file", False, "Whether to load model from file.")
tf.flags.DEFINE_boolean("save_model_to_file", False, "Whether to save model to file.")
tf.flags.DEFINE_integer("episodes_per_save", 10, "How often to save model to file.")
tf.flags.DEFINE_integer("seed", 1, "Seed for RNG.")
tf.flags.DEFINE_float("epsilon", 1.0, "Initial epsilon for exploration.")
tf.flags.DEFINE_float("epsilon_min", 0.1, "Minimal epsilon for exploration.")
tf.flags.DEFINE_float("epsilon_decay", 0.5, "Log decay for epsilon.")


NUM_EPISODES = 300
MAX_TIMESTEPS = 300
BATCH_SIZE = 32
GAMMA = .9
LEARNING_RATE = .01
TAU = .01
TARGET_ACTOR_FILE = "saved_models/target_actor.h5"
TARGET_CRITIC_FILE = "saved_models/target_critic.h5"
ACTOR_FILE = "saved_models/actor.h5"
CRITIC_FILE = "saved_models/critic.h5"


class DDPGTrainer(object):
    def __init__(self, env):
        self._buffer = replay_buffer.ReplayBuffer()
        self.env = env
        (self.critic_state_input, self.action_input, self.critic) = self._create_critic_model()
        [self.variable_summaries(x, "critic") for x in self.critic.trainable_weights]
        (self.actor_state_input, self.actor) = self._create_actor_model()
        [self.variable_summaries(x, "actor") for x in self.actor.trainable_weights]
        self.target_critic = Model.from_config(self.critic.get_config())
        self.target_critic.set_weights(self.critic.get_weights())
        [self.variable_summaries(x, "target_critic") for x in 
         self.target_critic.trainable_weights]
        self.target_actor = Model.from_config(self.actor.get_config())
        self.target_actor.set_weights(self.actor.get_weights())
        [self.variable_summaries(x, "target_actor") for x in 
         self.target_actor.trainable_weights]
        if FLAGS.load_model_from_file:
          self.critic.load_model(TARGET_CRITIC_FILE)
          self.actor.load_model(TARGET_ACTOR_FILE)
          self.critic.load_model(CRITIC_FILE)
          self.actor.load_model(ACTOR_FILE)
        self.epsilon_min = FLAGS.epsilon_min
        self.epsilon = FLAGS.epsilon
        self.epsilon_decay = FLAGS.epsilon_decay
        self.assign_weights = self.get_weight_assignment_op()
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.summarize = None
        self.train_writer = tf.summary.FileWriter(
            "train_summaries/train_%d" % time.time(), self.sess.graph)
        self.action_grads =  K.gradients(self.critic.outputs[0] / BATCH_SIZE,
            self.action_input)
        self.sess.run(tf.global_variables_initializer())

    def _create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='sigmoid')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='sigmoid')(merged)
        output = Dense(1)(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output)

        adam = tf.train.AdamOptimizer(LEARNING_RATE)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def generate_action(self, actor_output, action_space):
        if type(action_space) == gym.spaces.Discrete:
            if np.random.random() <= self.epsilon:
              return self.env.action_space.sample()
            else:
              return np.argmax(actor_output)
        return 0

    def _create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24)(state_input)
        output = Dense(self.env.action_space.shape[0], activation='softmax')(h1)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)

        self._action_gradient = tf.placeholder(
            tf.float32, [None, self.env.action_space.shape[0]])
        weights = model.trainable_weights
        grads = tf.gradients(output, weights, -self._action_gradient)
        adam = tf.train.AdamOptimizer(.000001)
        self._optimize_actor = adam.apply_gradients(zip(grads, weights))
        return state_input, model

    def eval_model(self):
        for episode in range(10):
          total_reward = 0.0
          cur_state = self.env.reset()
          done = False
          i = 0
          while not done and i < MAX_TIMESTEPS:
              actor_output = self.actor.predict(np.expand_dims(cur_state, 0))[0]
              action = self.generate_action(actor_output, self.env.action_space)
              self.env.render()
              observation, reward, done, info = self.env.step(action)
              total_reward += reward
              cur_state = observation
          print("EPISODE: %d NUM_STEPS: %d TOTAL_REWARD: %f" % (episode, i, total_reward))
          raw_input("Continue?")


    def variable_summaries(self, var, suffix):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries_%s' % suffix):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def train_model(self):
        t = 0
        for episode in range(NUM_EPISODES):
            total_reward = 0.0
            cur_state = self.env.reset()
            done = False
            i = 0
            while not done and i < MAX_TIMESTEPS:
                t += 1
                actor_output = self.actor.predict(np.expand_dims(cur_state, 0))[0]
                action = self.generate_action(actor_output, self.env.action_space)
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                self._buffer.add(cur_state, actor_output, reward, int(not done), observation)
                cur_state = observation
                if self._buffer.size() > BATCH_SIZE:
                    old_states, actions, rewards, not_dones, new_states = self._buffer.sample(
                        BATCH_SIZE)
                    predicted_actions = self.target_actor.predict(new_states)
                    sample_q = self.target_critic.predict_on_batch([new_states, predicted_actions])
                    sample_y = rewards + GAMMA * sample_q * not_dones
                    # print(self.sess.run([mean_target, mean]))
                    if self.summarize is None:
                      [self.variable_summaries(x, "predicted_actions")
                       for x in self.target_actor.outputs]
                      self.variable_summaries(self.target_critic.outputs[0], "sample_q")
                      self.summarize = tf.summary.merge_all()
                      reward_ph = tf.placeholder(tf.int32, shape=())
                      self.reward_summary = (tf.summary.scalar("total_reward", reward_ph), reward_ph)
                    if self.summarize is not None:
                      self.train_writer.add_summary(
                          self.sess.run(self.summarize, feed_dict={
                            self.target_actor.inputs[0]:new_states,
                            self.target_critic.inputs[0]:new_states,
                            self.target_critic.inputs[1]:predicted_actions
                          }), t)
                    self.critic.train_on_batch(x=[old_states, actions], y=sample_y)
                    action_grads = self.sess.run(self.action_grads,
                                                 feed_dict={self.action_input: actions,
                                                            self.critic_state_input:old_states})
                    optimize_weights = self.sess.run(
                        self._optimize_actor,
                        feed_dict={self._action_gradient:action_grads[0],
                        self.actor_state_input:old_states})
                    self.sess.run(self.assign_weights)
            if self.summarize is not None:
              self.train_writer.add_summary(
                  self.sess.run(self.reward_summary[0], feed_dict={
                      self.reward_summary[1]:total_reward}), t)
            self.epsilon = self.get_epsilon(t)
            print("EPISODE: %d NUM_STEPS: %d TOTAL_REWARD: %f" % (episode, i, total_reward))
            if FLAGS.save_model_to_file and episode % FLAGS.episodes_per_save == 0:
              self.actor.save(ACTOR_FILE)
              self.target_actor.save(TARGET_ACTOR_FILE)
              self.critic.save(CRITIC_FILE)
              self.target_critic.save(TARGET_CRITIC_FILE)

    def get_epsilon(self, t):
        return max(
            self.epsilon_min, min(self.epsilon, 1.0 -
                                  math.log10((t + 1) * self.epsilon_decay)))

    def get_weight_assignment_op(self):
        assign_weights = []
        target_actor_weights = self.target_actor.trainable_weights
        actor_weights = self.actor.trainable_weights
        for i in range(len(actor_weights)):
            assign_weights.append(target_actor_weights[i].assign(TAU * actor_weights[i] + (1 - TAU) * target_actor_weights[i]))
        target_critic_weights = self.target_critic.trainable_weights
        critic_weights = self.critic.trainable_weights
        for i in range(len(critic_weights)):
            assign_weights.append(target_critic_weights[i].assign(TAU * critic_weights[i] + (1 - TAU) * target_critic_weights[i]))
        print(assign_weights)
        return assign_weights

def main(argv):
  tf.set_random_seed(FLAGS.seed)
  env = gym.make('CartPole-v0')
  t = DDPGTrainer(env)
  t.train_model()
  t.eval_model()

if __name__ == "__main__":
  tf.app.run()
