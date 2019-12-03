# KOLON Benit Deep Learning Examples
- CDSW Hands-On Course
- Date : 2019.12.04
- Contact me : injoyer@yonsei.ac.kr

## TensorFlow Examples

This tutorial was designed for easily diving into TensorFlow, through examples. For readability, it includes both notebooks and source codes with explanation, for both TF v1 & v2.

It is suitable for beginners who want to find clear and concise examples about TensorFlow. Besides the traditional 'raw' TensorFlow implementations, you can also find the latest TensorFlow API practices (such as `layers`, `estimator`, `dataset`, ...).

If you have a question, feel free to email me (injoyer@yonsei.ac.kr)

## Tutorial index

#### 0 - Prerequisite
- [Introduction to Machine Learning](https://github.com/joyer7/Prerequisite/ml_introduction.ipynb).
- [Introduction to MNIST Dataset](https://github.com/joyer7/Prerequisite/mnist_dataset_intro.ipynb).

#### 1 - Introduction
- **Hello World** ([code](https://github.com/joyer7/tensorflow/helloworld.py)). Very simple example to learn how to print "hello world" using TensorFlow.
- **Basic Operations** ([code](https://github.com/joyer7/tensorflow/basic_operations.py)). A simple example that cover TensorFlow basic operations.
- **TensorFlow Eager API basics** ([code](https://github.com/joyer7/tensorflow/basic_eager_api.py)). Get started with TensorFlow's Eager API.

#### 2 - Basic Models
- **Linear Regression**  ([code](https://github.com/joyer7/tensorflow/linear_regression.py)). Implement a Linear Regression with TensorFlow.
- **Linear Regression (eager api)** ([code](https://github.com/joyer7/tensorflow/linear_regression_eager_api.py)). Implement a Linear Regression using TensorFlow's Eager API.
- **Logistic Regression**  ([code](https://github.com/joyer7/tensorflow/logistic_regression.py)). Implement a Logistic Regression with TensorFlow.
- **Logistic Regression (eager api)**  ([code](https://github.com/joyer7/tensorflow/logistic_regression_eager_api.py)). Implement a Logistic Regression using TensorFlow's Eager API.
- **Nearest Neighbor** ([code](https://github.com/joyer7/tensorflow/nearest_neighbor.py)). Implement Nearest Neighbor algorithm with TensorFlow.
- **K-Means**  ([code](https://github.com/joyer7/tensorflow/kmeans.py)). Build a K-Means classifier with TensorFlow.
- **Random Forest**   ([code](https://github.com/joyer7/tensorflow/random_forest.py)). Build a Random Forest classifier with TensorFlow.
- **Gradient Boosted Decision Tree (GBDT)**  ([code](https://github.com/joyer7/tensorflow/gradient_boosted_decision_tree.py)). Build a Gradient Boosted Decision Tree (GBDT) with TensorFlow.
- **Word2Vec (Word Embedding)** ([code](https://github.com/joyer7/tensorflow/word2vec.py)). Build a Word Embedding Model (Word2Vec) from Wikipedia data, with TensorFlow.

## TensorFlow 2.0

The tutorial index for TF v2 is available here: [TensorFlow 2.0 Examples](tensorflow_v2).

## Dataset
Some examples require MNIST dataset for training and testing. Don't worry, this dataset will automatically be downloaded when running examples.
MNIST is a database of handwritten digits, for a quick description of that dataset, you can check [this notebook](https://github.com/joyer7/Prerequisite/mnist_dataset_intro.ipynb).

Official Website: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).

## Installation

To download all the examples, simply clone this repository:
```
git clone https://github.com/joyer/cdsw
```

To run them, you also need the latest version of TensorFlow. To install it:
```
pip install tensorflow
```

or (with GPU support):
```
pip install tensorflow_gpu
```

For more details about TensorFlow installation, you can check [TensorFlow Installation Guide](https://www.tensorflow.org/install/)

## More Examples
The following examples are coming from [TFLearn](https://github.com/tflearn/tflearn), a library that provides a simplified interface for TensorFlow. You can have a look, there are many [examples](https://github.com/tflearn/tflearn/tree/master/examples) and [pre-built operations and layers](http://tflearn.org/doc_index/#api).

### Tutorials
- [TFLearn Quickstart](https://github.com/tflearn/tflearn/blob/master/tutorials/intro/quickstart.md). Learn the basics of TFLearn through a concrete machine learning task. Build and train a deep neural network classifier.

### Examples
- [TFLearn Examples](https://github.com/tflearn/tflearn/blob/master/examples). A large collection of examples using TFLearn.
