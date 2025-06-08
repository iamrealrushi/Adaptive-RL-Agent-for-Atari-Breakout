import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
import cv2
from collections import deque
import random

# ==== DQN Model ====
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.out = tf.keras.layers.Dense(num_actions)

    def call(self, x):
        x = tf.cast(x, tf.float32) / 255.0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)

# ==== Frame stack & preprocessing ====
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
    stacked_state = np.stack(frame_stack, axis=-1)
    return stacked_state

def main():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    num_actions = env.action_space.n
    model = DQN(num_actions)
    model.build(input_shape=(None, 84, 84, 4))
    model.load_weights("dqn_breakout_weights.weights.h5")  # Adjust filename as needed

    s_rgb, _ = env.reset()
    frame_stack.clear()
    state = get_stacked_state(s_rgb)
    done = False
    total_reward = 0
    EPSILON = 0.1  # 10% random actions during evaluation

    while not done:
        cv2.imshow("Breakout", cv2.resize(s_rgb, (420, 420)))
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            q_vals = model(np.expand_dims(state, axis=0), training=False)
            q_vals_np = q_vals.numpy()[0]
            print(f"Q-values: {q_vals_np}")
            action = np.argmax(q_vals_np)

        s_rgb, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = get_stacked_state(s_rgb)
        total_reward += reward
        print(f"Action: {action}, Reward: {reward}")

    print(f"Evaluation Finished | Total Reward: {total_reward}")
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()
