# Install required packages
# !pip install numpy pandas matplotlib tensorflow keras gym opencv-python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces
import cv2
import random
from collections import deque
import os
import zipfile
from google.colab import drive

# Mount Google Drive (specific to Colab; comment out if running locally)
drive.mount('/content/drive')

# Define paths
DATASET_ZIP_PATH = '/content/drive/MyDrive/archive.zip'
EXTRACTED_DIR = '/content/drive/MyDrive/breakhis'
IMAGE_DIR = f'{EXTRACTED_DIR}/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'

# Unzip the dataset
with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACTED_DIR)

# Function to load a small subset of images and labels
def load_image_subset(directory, sample_size=10):
    """Load and preprocess a small subset of images from the dataset."""
    image_list = []
    label_list = []
    for root, _, files in os.walk(directory):
        png_files = [f for f in files if f.lower().endswith('.png')]
        random.shuffle(png_files)
        for filename in png_files[:sample_size]:
            filepath = os.path.join(root, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))
            image_list.append(image)
            label_list.append(0 if 'benign' in filepath.lower() else 1)
        if len(image_list) >= sample_size:
            break
    return np.array(image_list), np.array(label_list)

# Load and preprocess data
images, labels = load_image_subset(IMAGE_DIR)
images = images.reshape(-1, 64, 64, 1) / 255.0  # Normalize pixel values

# Custom Gym environment for cancer image classification
class CancerClassificationEnv(gym.Env):
    """Environment for RL-based cancer image classification."""
    def __init__(self, images, labels):
        super(CancerClassificationEnv, self).__init__()
        self.images = images
        self.labels = labels
        self.step_index = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(64, 64, 1), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 = benign, 1 = malignant

    def reset(self):
        self.step_index = 0
        return self.images[self.step_index]

    def step(self, action):
        reward = 1 if action == self.labels[self.step_index] else -1
        self.step_index += 1
        is_done = self.step_index >= len(self.images)
        next_observation = self.images[self.step_index] if not is_done else np.zeros((64, 64, 1))
        return next_observation, reward, is_done, {}

    def render(self, mode='human'):
        pass  # No rendering implemented

# Initialize environment
env = CancerClassificationEnv(images, labels)

# Build the DQN model
def create_dqn_model(input_shape, num_actions):
    """Create a convolutional DQN model for image classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(24, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

INPUT_SHAPE = (64, 64, 1)
NUM_ACTIONS = env.action_space.n
dqn_model = create_dqn_model(INPUT_SHAPE, NUM_ACTIONS)

# Train the DQN agent
def train_dqn_agent(environment, model, num_episodes=2, batch_size=2):
    """Train the DQN model using reinforcement learning."""
    replay_buffer = deque(maxlen=50)  # Small memory buffer for experience replay
    discount_factor = 0.95  # Gamma for future rewards

    for episode in range(num_episodes):
        current_state = environment.reset()
        current_state = np.reshape(current_state, [1, 64, 64, 1])
        for step in range(10):  # Limited steps per episode
            action = np.argmax(model.predict(current_state, verbose=0)[0])
            next_state, reward, done, _ = environment.step(action)
            next_state = np.reshape(next_state, [1, 64, 64, 1])
            replay_buffer.append((current_state, action, reward, next_state, done))
            current_state = next_state

            if done:
                print(f"Episode: {episode}/{num_episodes}, Steps: {step}, Exploration: 1.0")
                break

            if len(replay_buffer) > batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                        target = reward + discount_factor * np.amax(model.predict(next_state, verbose=0)[0])
                    target_values = model.predict(state, verbose=0)
                    target_values[0][action] = target
                    model.fit(state, target_values, epochs=1, verbose=0)

train_dqn_agent(env, dqn_model)

# Evaluate the trained agent
def evaluate_agent(environment, model, num_episodes=10):
    """Evaluate the agent's performance over multiple episodes."""
    total_reward = 0
    for episode in range(num_episodes):
        current_state = environment.reset()
        current_state = np.reshape(current_state, [1, 64, 64, 1])
        done = False
        while not done:
            action = np.argmax(model.predict(current_state, verbose=0)[0])
            next_state, reward, done, _ = environment.step(action)
            current_state = np.reshape(next_state, [1, 64, 64, 1])
            total_reward += reward
    average_reward = total_reward / num_episodes
    print(f"Average Reward: {average_reward}")

evaluate_agent(env, dqn_model)

# Preprocess a single test image
def preprocess_test_image(image_path):
    """Load and preprocess a single test image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    image = image.reshape(-1, 64, 64, 1) / 255.0
    return image

# Test prediction on a sample image
TEST_IMAGE_PATH = (
    f'{IMAGE_DIR}/benign/SOB/adenosis/SOB_B_A_14-22549AB/100X/'
    'SOB_B_A-14-22549AB-100-001.png'
)
test_image = preprocess_test_image(TEST_IMAGE_PATH)

# Make a prediction
prediction = dqn_model.predict(test_image, verbose=0)
predicted_class = np.argmax(prediction)
class_labels = {0: 'benign', 1: 'malignant'}
predicted_label = class_labels[predicted_class]
print(f"Predicted Label: {predicted_label}")