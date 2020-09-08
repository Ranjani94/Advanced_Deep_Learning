
## Multiple Instance Learning

It is defined as weakly supervised learning algorithm where training data are split into two bags which is positive and negative bags and one label for each bag. All the instances in the bag is said to be positive if any one instance is positive whereas if only all of the instances are negative the bag is negative. Label for individual instances exist inside bag but are unknown suring the training. 

### MNIST Dataset

MNIST dataset is a collection of handwritten digits images which has label from 0 to 9. This dataset has 60,000 training set and 10,000 test set. Loading the dataset using pytorch library and defining the batch size as 256 for both training and test. The dataset is resized to 224 X 224 pixel size and normalized by taking mean and standard deviation. Data is split into training and validation data.

### Pre-trained ResNet Model

First step is to train the pre trained ResNet model using the labeled dataset that is loaded. Second, will pass the bag labeled dataset to the same model so that we can extra the features. Ant at last we will apply the Multiple Instance Learning model on the dataset.

The Residual bloack has convolutional two dimensional with kernel size 7, stride as (2,2) and paddig as (3,3)

Evalution metrices funtion is used to calculate the difference between true and predicted output values from the model. Scores like precision, recall, F1, accuracy is calculated to evaluate the model performance.

Training the model and calculating the losses, training loss and validation loss, evaluation metrices and batches

### Multiple Instance Learning

Generate bag labels, bag indices, bag features from the extracted feature set from labeled dataset. Since our bag has different sizes, we need to pad each tensor to have the same shape (max: 7). We will look through each one instance and look at the shape of the tensor, and then we will pad 7-n to the existing tensor where n is the number of instances in the bag. The function will return a padded data set. Defining the model and used LeakyReLu as an activation function, dropout for the middle layer and softmax for the final layer.
