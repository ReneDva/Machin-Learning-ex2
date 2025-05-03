# MNIST Classification with Custom Neural Network

![Machine Learning](https://img.shields.io/badge/Machine_Learning-Project-blueviolet)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00)
![Accuracy](https://img.shields.io/badge/Accuracy-98.25%25-brightgreen)
![Experiments](https://img.shields.io/badge/Experiments-15+-important)

**Student**: Rene Dvash  
**Academic Project Requirement**: Network architecture based on personal ID `xxxxxxxxx`

## 🔍 Project Overview
This project implements a neural network for MNIST digit classification, with a unique architecture derived from the digits of my ID number. Key features:

- Custom layer sizes calculated from ID digits
- Comprehensive hyperparameter testing (30 configurations)
- Achieved **96.72% test accuracy**
- Detailed experiment documentation

## 🧮 ID-Based Architecture
Per project requirements, layer sizes were determined by my ID `xxxxxxxxx`:

| Layer      | Calculation (ID: xxxxxxxxx) | Neurons | Implementation |
|------------|-----------------------------|---------|----------------|
| Input      | -                           | 784 (28×28) | `Flatten()` |
| Hidden 1   | Adjacent pair 1: 00 → 21   | 10      | `Dense(10)` + `BatchNorm` + `ReLU` |
| Hidden 2   | Adjacent pair 2: 82 → 82   | 82      | `Dense(82)` + `BatchNorm` + `ReLU` |
| Hidden 3   | Adjacent pair 3: 46 → 46   | 46      | `Dense(46)` + `BatchNorm` + `ReLU` |
| Hidden 4   | Adjacent pair 4: 13 → 13   | 13      | `Dense(13)` + `BatchNorm` + `ReLU` |
| Output     | -                           | 10      | `Dense(10)` + `Softmax()` |

```python
layers = [
tf.keras.layers.Flatten(input_shape=image_shape),

tf.keras.layers.Dense(10),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),

tf.keras.layers.Dense(82),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),

tf.keras.layers.Dense(46),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),

tf.keras.layers.Dense(13),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),

tf.keras.layers.Dense(num_of_classes),
tf.keras.layers.Softmax()
  ]
```
**Results:**
Google sheet containing the experiments and Model evaluation per experiment.
[(https://docs.google.com/spreadsheets/d/1nGdS58SeRuDPwSIJWjXA5pTFu1Fe6aVX8yT0cp7K-w4/edit?usp=sharing)](https://docs.google.com/spreadsheets/d/1nGdS58SeRuDPwSIJWjXA5pTFu1Fe6aVX8yT0cp7K-w4/edit?usp=sharing)

