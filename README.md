# cnn_activity_recognition
Human activity recognition on raw signal data from UCI HAR dataset using cnn in pytorch

This model uses cnn architecture to classify the activity being performed by humans.  The data is from the UCI machine 
 learning repository.  The dataset used is human activity recognition using smartphones.
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

For the cnn model, I used the first 6 channels of unprocessed data from this dataset, which are x, y and z body acceleration
and rotation recorded using a smartphone accelerometer and gyroscope, respectively.  I ommitted the total acceleration from
use in the model, because body acceleration is the same reading after accounting for the acceleration due to the force of 
gravity.

This model achieved 92.0% validation set accuracy and 83.7% test set accuracy, in contrast to the 96% accuracy achieved 
when using linear dirscriminant analysis on the set of processed features also available as part of the dataset.

Goals to add to this notebook:
1) add label support for dataloaders so that I can create a confusion matrix that displays acitivities
2) Visualize the training and validation performance on tensorboard
3) Consider using stratified train test split to split all three dets for possible improved performance.
4) Normalize the incoming data through a transform
5) Clean up the pipeline so that all transforms happen using the transform method
