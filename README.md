# MNIST Classification with Custom Neural Network

![Machine Learning](https://img.shields.io/badge/Machine_Learning-Project-blueviolet)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00)
![Accuracy](https://img.shields.io/badge/Accuracy-98.25%25-brightgreen)
![Experiments](https://img.shields.io/badge/Experiments-15+-important)

**Student**: Rene Dvash  
**Academic Project Requirement**: Network architecture based on personal ID `213468200`

## 🔍 Project Overview
This project implements a neural network for MNIST digit classification, with a unique architecture derived from the digits of my ID number. Key features:

- Custom layer sizes calculated from ID digits
- Comprehensive hyperparameter testing (15+ configurations)
- Achieved **98.25% test accuracy**
- Detailed experiment documentation

## 🧮 ID-Based Architecture
Per project requirements, layer sizes were determined by my ID `213468200`:

| Layer      | Calculation (ID: 213468200) | Neurons | Implementation |
|------------|-----------------------------|---------|----------------|
| Input      | -                           | 784 (28×28) | `Flatten()` |
| Hidden 1   | Adjacent pair 1: 2,1 → 21   | 21      | `Dense(21)` + `BatchNorm` + `ReLU` |
| Hidden 2   | Adjacent pair 2: 1,3 → 13   | 13      | `Dense(13)` + `BatchNorm` + `ReLU` |
| Hidden 3   | Adjacent pair 3: 3,4 → 34   | 34      | `Dense(34)` + `BatchNorm` + `ReLU` |
| Hidden 4   | Adjacent pair 4: 4,6 → 46   | 46      | `Dense(46)` + `BatchNorm` + `ReLU` |
| Output     | -                           | 10      | `Dense(10)` + `Softmax()` |

```python
layers = [
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(21),  # זוג ראשון: 21
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(13),  # זוג שני: 13
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(34),  # זוג שלישי: 34
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(46),  # זוג רביעי: 46
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(10),  # Softmax output
    tf.keras.layers.Softmax()
]
```
**Results:**
Google sheet containing the experiments and Model evaluation per experiment.
[(https://docs.google.com/spreadsheets/d/1nGdS58SeRuDPwSIJWjXA5pTFu1Fe6aVX8yT0cp7K-w4/edit?usp=sharing)](https://docs.google.com/spreadsheets/d/1nGdS58SeRuDPwSIJWjXA5pTFu1Fe6aVX8yT0cp7K-w4/edit?usp=sharing)

