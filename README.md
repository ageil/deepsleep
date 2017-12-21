# DeepSleep

This is a repository for the DeepSleep project, developed during the course 02456 Deep Learning at DTU (Fall, 2017).

The repository contains:

- A [**demo**](https://github.com/ageil/deepsleep/blob/master/demo/CRNN%20DEMO.ipynb) folder with a Jupyter Notebook implementing a minimal working example of 1) our CRNN training loop and 2) a prediction example using our own pretrained weights.
- A [**visualization**](https://github.com/ageil/deepsleep/blob/master/visualization/keras-vis.ipynb) Jupyter Notebook investigating the internals of our models, and showcasing some of the plots used in the presentation and report (bar plots, saliency maps, etc).
- The raw scripts and data batching procedures used for the actual training sessions.

The contents were carried out and tested on MacOS 10.13.2 with Python 3.6.2, using TensorFlow 1.3.0, Keras 2.0.8, Numpy 1.13.3 and Matplotlib 2.0.2. 
Please note that the visualization notebook in particular does not work with the latest Keras implementation (2.1.2).
