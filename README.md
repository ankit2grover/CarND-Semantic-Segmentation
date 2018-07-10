# UDACITY SELF-DRIVING CAR ENGINEER NANODEGREE
# Semantic Segmentation Project (Advanced Deep Learning)

## Introduction

The goal of this project is to construct a fully convolutional neural network based on the VGG-16 image classifier architecture for performing semantic segmentation to identify drivable road area from an car dashcam image (trained and tested on the KITTI data set).

## Source Code
All the source code is in mainJupyter.pynb file that I have used in AWS for training and predictions. Augument source code is in augument.py.
Also, helper code also has been modified and that is in helper.py
I have also updated main.py file that is in sync with mainJupyter.pynb to train the model and run inference on the same model for test images.

## Saved Models and Inference Results
All the saved models are in savedModel folder and test images results are in run folder.

## Approach

### Architecture

A pre-trained VGG-16 network was converted to a fully convolutional network by converting the final fully connected layer to a 1x1 convolution and setting the depth equal to the number of desired classes (in this case, two: road and not-road). Performance is improved through the use of skip connections, performing 1x1 convolutions on previous VGG layers (in this case, layers 3 and 4) and adding them element-wise to upsampled (through transposed convolution) lower-level layers (i.e. the 1x1-convolved layer 7 is upsampled before being added to the 1x1-convolved layer 4). Each convolution and transpose convolution layer includes a kernel initializer and regularizer

### Optimizer

The loss function for the network is cross-entropy, and an Adam optimizer is used.

### Training

The hyperparameters used for training are:

  - keep_prob: 0.5
  - learning_rate: 0.0009
  - epochs: 50
  - batch_size: 5

### Augumentation
 Augumented every image also to to improve the results
  
  - Converted RGB image into HSV and changed brightness randomly by 20%-30%
  - Rotated the image randomly between values 0 to 45 degree.
  - Translated the image randomly between 0 to 20.
  - Shear translate the image between 0 to 20.

  

## Results

### Without Augumentation

Loss per batch tends to average below 0.58700 after two epochs and below 0.280 after ten epochs. Average loss per batch at epoch 20: 0.145, at epoch 30: 0.094, at epoch 40: 0.053, and at epoch 50: 0.044.

### With Augumentation

Loss per batch tends to average below 10.698 after two epochs and below 0.66 after ten epochs. Average loss per batch at epoch 20: 0.35, at epoch 40: 0.275.

### Conclusion
Augumentation helps but here we are getting better results without augumentation. I will consider as a improvement that try some other values of augumentation so that entropy loss can be decreased further.

### Samples

Below are a few sample images from the output of the fully convolutional network, with the segmentation class overlaid upon the original image in green.

![sample1](./resultSamples/sample1.png)
![sample2](./resultSamples/sample2.png)
![sample3](./resultSamples/sample3.png)
![sample4](./resultSamples/sample4.png)
![sample5](./resultSamples/sample5.png)

Performance is very good, but not perfect with only spots of road identified in a handful of images.


---

## *The following is from the original Udacity repository README*

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).