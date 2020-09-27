### SimCLR: A Simple Framework for Contrastive Learning of Visual Representation

A massive number of unannotate images are produced in the current world and the cost to annotate them is so expensive. To overcome this problem, SimCLR is introduced where the good representations are learned from the unlabeled data instead of supervised data. This would definitely help to extract the downstream task which has limited annotated data and also proved state-of-the-art accuracy. The idea of Contrastive learning to learn the representations while enforcing similar elements to be equal and dissimilar elements to be different.

SimCLR uses ResNet50 pre trained model as a backbone. The ResNet model is fed with augmented images of shape (224,224,3) and it outputs a 2048 dimensional embeddign vector. A Multilayer perceptron with two dense layers is applied to the embedding vectoe which produces the final representation. After the training, we can delete the projection head whihc is MLP and use only the representations to learn the new downstream task.

Reference: https://colab.research.google.com/github/google-research/simclr/blob/master/colabs/finetuning.ipynb#scrollTo=BxhfMmVdHoZM
