
# Reinforcement Learning for Enhanced Medical Imaging


*Enhancing cancer diagnostics through reinforcement learning and computer vision.*

## Overview

This project implements a **Deep Q-Network (DQN)** to classify breast cancer histology images as benign or malignant using reinforcement learning (RL). Leveraging the **BreaKHis dataset**, the model processes grayscale images with OpenCV, trains a convolutional neural network (CNN) via TensorFlow/Keras, and optimizes decisions in a custom Gym environment. The goal is to explore RL's potential in medical imaging, aiming to improve diagnostic accuracy and efficiency.

Key features:

- Preprocesses 64x64 grayscale images from the BreaKHis dataset.

- Uses a custom Gym environment to simulate cancer classification tasks.

- Trains a DQN agent with experience replay for robust learning.

- Evaluates performance and predicts on sample images.

This work aligns with my CV project "Reinforcement Learning for Enhanced Medical Image Classification," demonstrating my expertise in Python, AI, and healthcare imaging solutions.

---

## Table of Contents

- [Prerequisites](#prerequisites)

- [Installation](#installation)

- [Dataset](#dataset)

- [Usage](#usage)

- [Project Structure](#project-structure)

- [How It Works](#how-it-works)

- [Future Improvements](#future-improvements)

---

## Prerequisites

- **Python 3.8+**

- **Google Colab** (optional, for Drive integration)

- Libraries:

  - `numpy`

  - `pandas`

  - `matplotlib`

  - `tensorflow`

  - `keras`

  - `gym`

  - `opencv-python`

---

## Installation

1\. **Clone the Repository**:

   `git clone https://github.com/Shreen-A/Reinforcement-Learning-for-Enhanced-Medical-Imaging.git`

   `cd Reinforcement-Learning-for-Enhanced-Medical-Imaging`

2\. **Install Required Packages**:

   Install the necessary Python libraries using pip:

   `pip install numpy pandas matplotlib tensorflow keras gym opencv-python`
   
3\. **Set Up Google Drive (Optional)**:

   - If using Google Colab, mount your Google Drive to access the dataset:

     `from google.colab import drive`

     `drive.mount('/content/drive')`

   - Place the BreaKHis dataset ZIP file (`archive.zip`) in `/content/drive/MyDrive/`.

   - Skip this step if running locally and adjust file paths accordingly.

4\. **Run Locally**:

   - Comment out Colab-specific lines (e.g., `drive.mount`) in `Medical_Image_RL.py`.

   - Ensure the dataset ZIP is in your local directory and update `DATASET_ZIP_PATH` to match.

---

## Dataset

The project uses the **[BreaKHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis)**, a collection of breast histology images:

- **Images**: Over 7,900 histology slides labeled as benign or malignant.

- **Structure**: Organized by tumor type (e.g., adenosis) and magnification (e.g., 100X).

- **Subset**: This demo uses a small sample (10 images) for quick training; expand for full potential.

### Preparing the Dataset

1\. Download `archive.zip` from Kaggle or your source.

2\. Upload it to Google Drive (`/content/drive/MyDrive/`) or your local directory.

3\. The script unzips it to `/content/drive/MyDrive/breakhis` and processes a subset.

---

## Usage

1\. **Run the Script**:

   - Open `Medical_Image_RL.py` in Google Colab or a local Python environment (e.g., Jupyter Notebook, VS Code).

   - Execute the script to:

     - Load and preprocess a subset of BreaKHis images.

     - Train the DQN agent over 2 episodes.

     - Evaluate performance across 10 episodes.

     - Predict the class of a sample test image.

2\. **Key Code Snippets**:

   - **Loading and Preprocessing Images**:

     `images, labels = load_image_subset(IMAGE_DIR, sample_size=10)`

     `images = images.reshape(-1, 64, 64, 1) / 255.0`
     

   - **Defining the DQN Model**:

     model = Sequential([

         Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),

         Flatten(),

         Dense(24, activation='relu'),

         Dense(2, activation='linear')

     ])

     model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

   
   - **Training the Agent**:

     `train_dqn_agent(env, dqn_model, num_episodes=2, batch_size=2)`

    
   - **Evaluating Performance**:

     evaluate_agent(env, dqn_model, num_episodes=10)


   - **Predicting on a Test Image**:

     test_image = preprocess_test_image(TEST_IMAGE_PATH)

     predicted_class = np.argmax(dqn_model.predict(test_image, verbose=0))

     print(f"Predicted Label: {class_labels[predicted_class]}")

     
3\. **Expected Output**:

   - Training logs: "Episode: 1/2, Steps: 9, Exploration: 1.0"

   - Evaluation: "Average Reward: 0.8" (example; varies with training extent).

   - Prediction: "Predicted Label: benign" (for the sample test image).

---

## Project Structure

```

Reinforcement-Learning-for-Enhanced-Medical-Imaging/

├── Medical_Image_RL.py          # Main script with DQN implementation

├── README.md                   # This documentation

├── archive.zip                 # BreaKHis dataset (not tracked in repo)

├── requirements.txt            # Dependency list

└── outputs/                    # (Optional) Folder for screenshots or logs

```

---

## How It Works

1\. **Data Preprocessing**:

   - Loads a subset of images from BreaKHis, resizes to 64x64 grayscale, and normalizes to [0, 1].

2\. **Custom Gym Environment**:

   - Defines `CancerClassificationEnv`:

     - **Observation**: 64x64x1 grayscale image.

     - **Actions**: 0 (benign) or 1 (malignant).

     - **Reward**: +1 for correct classification, -1 for incorrect.

3\. **DQN Model**:

   - A CNN with a 32-filter convolutional layer, flattened, and dense layers (24, 2) predicts Q-values.

   - Compiled with MSE loss and Adam optimizer (learning rate 0.001).

4\. **Training**:

   - Uses experience replay (buffer size 50) and a discount factor of 0.95.

   - Trains over 2 episodes with limited steps (10) for this demo.

5\. **Evaluation and Prediction**:

   - Evaluates over 10 episodes to compute average reward.

   - Predicts on a single test image to demonstrate real-world application.


## Future Improvements

- **Scale Dataset**: Train on the full BreaKHis dataset (7,900+ images) for higher accuracy.

- **Tune Hyperparameters**: Optimize learning rate, buffer size, or episode count.

- **Advanced RL**: Explore Double DQN or Dueling DQN for better stability.

- **Metrics**: Add precision, recall, and F1-score for comprehensive evaluation.

- **Deployment**: Integrate with a UI (e.g., Flask) for interactive predictions.




