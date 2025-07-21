# Neural Network from Scratch

This project is a complete neural network framework built from scratch in pure Python + NumPy, with no external machine learning libraries. It supports building and training feedforward neural networks, including support for backpropagation, multiple optimizers, loss functions, regularization techniques, and prediction modes.

Tested on Fashion MNIST using raw pixel data, this project demonstrates how fundamental machine learning operations can be built from the ground up — including batching, regularization, dropout, prediction, and more.

---

## ✨ Features

### 🔧 Core Components

* Modular neural network architecture
* Fully connected (dense) layers
* Dropout layers for regularization
* Input layer abstraction
* Batch-based prediction support

### 🔊 Activation Functions

* ReLU (Rectified Linear Unit)
* Softmax
* Sigmoid
* Linear (identity)

### 📈 Loss Functions

* Categorical Cross-Entropy
* Binary Cross-Entropy
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Softmax + Cross-Entropy combined class for numerical stability and efficiency

### ⚖️ Regularization

* L1 and L2 regularization for both weights and biases
* Dropout support with proper scaling for inference

### ⚙️ Optimizers

* Stochastic Gradient Descent (SGD) with optional Momentum and Decay
* AdaGrad
* RMSProp
* Adam (Adaptive Moment Estimation)

### 🔄 Training Strategy

* Full-batch and mini-batch training
* Epoch-based training loop
* Batch shuffling and accumulation
* Real-time reporting of accuracy, loss, and learning rate
* Integrated validation support

### 🔍 Evaluation and Prediction

* Validation mode with accumulated metrics
* Batch-safe prediction mode
* Accuracy classes for classification and regression
* Forward-only inference pipeline for deployment

### 📂 Model Persistence

* Save/load model parameters
* Save/load entire model using `pickle`
* `get_parameters` and `set_parameters` interfaces

---

## 🔢 Example Usage

```python
# Define model
model = Model()
model.add(Layer_Dense(784, 128))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Compile
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.001),
    accuracy='categorical'
)
model.finalise()

# Train
model.train(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save
model.save("fashion_model.nn")

# Predict
predictions = model.predict(X_test)
```

---

## 📖 Example Scripts in This Repo

* `Neural_Networks.py`: Core model architecture, layers, activations, losses, optimizers, training engine
* `Training for Fashion MNIST.py`: Loads and trains a model using the Fashion MNIST dataset
* `Predictions for Fashion MNIST.ipynb`: Loads the trained model and runs predictions on new images

---

## 🚀 Potential Extensions

* [ ] Add Convolutional Layers
* [ ] Load/export to ONNX or TorchScript
* [ ] Add model visualizations and live loss plotting

Thanks!
