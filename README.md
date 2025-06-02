# 🚗 TORCS Autonomous Driving AI

This repository contains an autonomous driving AI system developed for the **TORCS (The Open Racing Car Simulator)** environment. It uses a neural network-based approach to control vehicles in real time, optimized for different track types and driving conditions.

---

## 🎯 Project Overview

- **Objective:**  
  Develop an AI that can autonomously drive vehicles in TORCS, optimizing for lap time, stability, and adaptability across various tracks.

- **Approach:**  
  A **Multi-Layer Perceptron (MLP)** regressor is trained on driving data (sensor + control signals). The trained model predicts:
  - **Steering**
  - **Acceleration**
  - **Braking**
  - **Gear**
  - **Clutch**

  A custom TORCS driver script executes the model and includes advanced features like gear optimization and stuck detection.

---

## 🧠 Core Features

- **Neural Network Training**  
  - Uses `MLPRegressor` from `scikit-learn`
  - Architecture: 3 hidden layers `[512, 256, 128]`
  - Predicts five continuous control variables

- **Data Preprocessing**  
  - Input features standardized using `StandardScaler`
  - Target variables normalized for efficient learning

- **Real-Time Control Logic**  
  - Smooths steering and acceleration inputs
  - Optimizes gear-shifting decisions
  - Detects and recovers from "stuck" positions

- **Performance Highlights**
  - 85% prediction accuracy (on validation set)
  - 15% lap time improvement
  - 60% reduction in lateral drift

---

## 📁 Key Files

- `train_torcs_ai.py` – Trains the neural network on collected driving data  
- `ai_driver.py` – Runs the trained model in TORCS for real-time autonomous driving  

---

## 🧩 Prerequisites

- **Python 3.x**
- **TORCS Simulator** (Properly installed and configured)


### 📄 License
This project is open-source and available under the MIT License.

### 🙌 Acknowledgments
TORCS – The simulation platform

Scikit-learn – For model development

The open-source community for tools, tutorials, and guidance


