import tensorflow as tf
import numpy as np

import replay_buffer


test  = tf.test
log = tf.logging
FLAGS = tf.flags.FLAGS


class ReplayBufferTest(test.TestCase):
  def testInit(self):
    with self.test_session():
      buffer = replay_buffer.ReplayBuffer()

  def testAdd(self):
    with self.test_session():
      buffer = replay_buffer.ReplayBuffer()
      for i in range(1000):
        buffer.add(np.array([i, 2*i]), np.array([i % 5,]), np.array([i]), False, np.array([2*i, 3*i]))
      self.assertEqual(1000, buffer.size())

  def testSample(self):
    with self.test_session():
      buffer = replay_buffer.ReplayBuffer()
      for i in range(1000):
        buffer.add(np.array([i, 2*i]), np.array([i % 5,]), np.array([i]), False, np.array([2*i, 3*i]))
      num_samples = 32
      for j in range(50):
        old_states, actions, rewards, dones, new_states = buffer.sample(num_samples)
        reward_set = set()
        for s in range(num_samples):
          i = rewards[s][0]
          self.assertNotIn(i, reward_set)
          reward_set.add(i)
          self.assertFalse(dones[s])
          self.assertTrue((actions[s] == i % 5).all())
          self.assertTrue((old_states[s] == np.array([i, 2*i])).all())
          self.assertTrue((new_states[s] == np.array([2*i, 3*i])).all())

  def testOverWrite(self):
    with self.test_session():
      buffer = replay_buffer.ReplayBuffer()
      for i in range(100000):
        buffer.add(np.array([i, 2*i]), np.array([i % 5,]), np.array([i]), False, np.array([2*i, 3*i]))
      num_samples = 32
      self.assertEqual(replay_buffer.MAX_SIZE , buffer.size())

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  test.main()
