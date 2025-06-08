import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
import random
from collections import deque
import cv2
import os
from tensorflow.keras import layers, optimizers, losses

# ==== DQN Model ====
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 8, strides=4, activation='relu')
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.out = layers.Dense(num_actions)

    def call(self, x):
        x = tf.cast(x, tf.float32) / 255.0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)

# ==== Preprocessing ====
frame_stack = deque(maxlen=4)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

def get_stacked_state(new_frame):
    processed = preprocess_frame(new_frame)
    frame_stack.append(processed)
    while len(frame_stack) < 4:
        frame_stack.append(processed)
    return np.stack(frame_stack, axis=-1)

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)

# ==== Dynamic Difficulty ====
def modify_environment_difficulty(env, episode, step_count):
    # Paddle speed simulation: delay actions
    if episode > 150 and step_count % 20 == 0:
        for _ in range(2):  # simulate slowdown
            env.step(0)  # NOOP

    # Ball speed: simulate speed spike
    if episode > 250 and step_count % 100 == 0:
        for _ in range(2):  # make ball move faster
            env.step(env.action_space.sample())

# ==== Hyperparameters ====
EPISODES = 400
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 100000
TARGET_UPDATE = 10
LEARNING_RATE = 1e-4
SAVE_PATH = "dqn_breakout_weights.weights.h5"

# ==== Main Training ====
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
num_actions = env.action_space.n

policy_net = DQN(num_actions)
target_net = DQN(num_actions)
target_net.set_weights(policy_net.get_weights())

optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = losses.Huber()
buffer = ReplayBuffer()

steps_done = 0

for ep in range(EPISODES):
    raw_frame, _ = env.reset()
    frame_stack.clear()
    state = get_stacked_state(raw_frame)
    done = False
    ep_reward = 0
    step = 0

    while not done:
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        if random.random() > eps:
            q = policy_net(np.expand_dims(state, 0))
            action = tf.argmax(q[0]).numpy()
        else:
            action = env.action_space.sample()

        modify_environment_difficulty(env, ep, step)
        next_raw, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        next_state = get_stacked_state(next_raw)

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        step += 1

        if len(buffer) >= BATCH_SIZE:
            s_b, a_b, r_b, s2_b, d_b = buffer.sample(BATCH_SIZE)

            with tf.GradientTape() as tape:
                q_vals = policy_net(s_b)
                idx = tf.stack([tf.range(BATCH_SIZE), a_b], axis=1)
                q_action = tf.gather_nd(q_vals, idx)

                q_next = target_net(s2_b)
                q_max = tf.reduce_max(q_next, axis=1)
                targets = r_b + GAMMA * q_max * (1 - d_b)
                loss = loss_fn(targets, q_action)

            grads = tape.gradient(loss, policy_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

    if ep % TARGET_UPDATE == 0:
        target_net.set_weights(policy_net.get_weights())

    print(f"Episode {ep} | Reward: {ep_reward} | Epsilon: {eps:.3f}")

# Save model
policy_net.save_weights(SAVE_PATH)
env.close()
