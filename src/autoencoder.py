import torch
from torch import nn            # import neural network building blocks
from torch.utils.data import Dataset, DataLoader  # used to manage input data and separate into batches
from torch import optim              # used to train the model parameters
import numpy as np
import os
import time
from process_dataset import standardWidth, standardHeight  # import input image dimensions
from process_dataset import processedDatasetPath           # import folder to save encoded images to
import cv2 as cv

# Set to false to disable cuda gpu
# acceleration.
# When true, it will only be used
# if the computer has an nvidia gpu
# and the installed pytorch version
# is built with support enabled.
# This should be enabled if possible
# for a massive training speedup.
# The gpu should have at least 3GB of
# vram to enable this option.
useCudaP = True
device = 'cuda' if useCudaP and torch.cuda.is_available() else 'cpu'
print("Cuda acceleration enabled and cuda support detected, will use cuda."  if device == 'cuda' else "Cuda either disabled or unavailable.")

# sets whether to rotate the training images
# to increase size of training set by a factor
# of 4
enhanceTrainingByRotation = True

def separateDatasets(percentTrain=0.9):  # use 90% of data for training by default since there isn't much data available
    """Splits the labels.txt dataset into train_labels.txt and test_labels.txt, for
    testing and training datasets, using the argument to determine how many images
    to use for training. This should be called after processDataset from
    process_dataset.py, since that function generates the labels.txt file."""
    inFile = open("labels.txt")
    lines = inFile.read().split('\n')
    imageNamesByLabel = {}
    # group the image names by label
    for line in lines:
        if line != "":
            imageName, labelString = line.split(',')
            label = int(labelString)
            if label in imageNamesByLabel:
                imageNamesByLabel[label].append(imageName)
            else:
                imageNamesByLabel[label] = [imageName]
    # split the images into training and testing data
    trainingNamesByLabel = {}
    testingNamesByLabel = {}
    for label in imageNamesByLabel:
        images = imageNamesByLabel[label]
        separatingIndex = int(len(images)*percentTrain)
        trainingImages = images[:separatingIndex]
        testingImages = images[separatingIndex:]
        trainingNamesByLabel[label] = trainingImages
        testingNamesByLabel[label] = testingImages
    # write the training and testing label files
    def writeLabelFile(path, namesByLabel):
        outFile = open(path, 'w')
        for label in namesByLabel:
            for name in namesByLabel[label]:
                outFile.write(name + ", " + str(label) + '\n')
    writeLabelFile("train_labels.txt", trainingNamesByLabel)
    writeLabelFile("test_labels.txt", testingNamesByLabel)

def loadImageToTensor(path):
    """Loads an image with opencv to a pytorch tensor.
    The returned tensor is 1-dimensional, for use with
    flat neural networks (not convolutional)."""
    # use opencv to read the grayscale image in numpy format,
    # then convert that array to a pytorch tensor vector
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    image = torch.from_numpy(image)  # loads the input image into a pytorch tensor
    # convert image matrix to float format
    image = image.to(dtype=torch.float)
    # convert the image from a matrix into a vector
    image = torch.flatten(image)

    # normalize image to 0-1 range
    image = torch.div(image, 255)
    # move image tensor to gpu if cuda is enabled and
    # supported on this computer, else do nothing
    image = image.to(device)
    return image

class ProcessedDataset(Dataset):
    def readLabels(self, labelFile):
        # read file lines
        inFile = open(labelFile)
        lines = inFile.read().split('\n')
        inFile.close()
        # record labels
        labels = {}
        for line in lines:
            # ignore empty line at the end of the document
            if line != "":
                fileName, labelString = line.split(',')
                labels[fileName] = int(labelString)
        return labels
    def readImages(self, labels):
        # for each label, load an image and maybe rotate
        # it 3 times to increase the amount of training data
        def cvImageToTensor(image):
            image = torch.from_numpy(image)  # loads the input image into a pytorch tensor
            # convert image matrix to float format
            image = image.to(dtype=torch.float)
            # convert the image from a matrix into a vector
            image = torch.flatten(image)
            # normalize image to 0-1 range
            image = torch.div(image, 255)
            # return the tensor image
            return image
        def allRotationsAsTensors(cv_image):
            r1 = cv_image
            r2 = np.rot90(r1, k=1, axes=(0,1)).copy()  # avoid stride errors by copying the array
            r3 = np.rot90(r2, k=1, axes=(0,1)).copy()
            r4 = np.rot90(r3, k=1, axes=(0,1)).copy()
            return [cvImageToTensor(x) for x in [r1, r2, r3, r4]]
        images = []
        for label in labels:
            imageName = label
            imagePath = os.path.join(self.directory, imageName)
            # read image in gray
            cvImage = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
            if enhanceTrainingByRotation:
                # rotate it each direction, and put those into data
                images.extend(allRotationsAsTensors(cvImage))
            else:
                images.append(cvImageToTensor(cvImage))
        return images

    def __init__(self, labelFile, directory, transform=None, target_transform=None):
        self.directory = directory
        self.transform = transform
        self.target_transform = target_transform
        # load all the dataset images in advance
        self.images = self.readImages(self.readLabels(labelFile))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # get the image at the given index
        image = self.images[index]
        # move the image to the target device
        image = image.to(device)
        
        # for autoencoders, the target output is the input itself, normalized to 0-1 range
        label = image
        if self.transform != None:
            image = self.transform(image)
        if self.target_transform != None:
            label = self.target_transform(label)

        return (image, label)

# define a pytorch api based neural network autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, encoderDimensions, decoderDimensions):
        super(Autoencoder, self).__init__()
        self.encoderDimensions = encoderDimensions  # possible example values: [2500, 800, 200, 10]
        self.decoderDimensions = decoderDimensions  # and [10, 200, 800, 2500]
        def layersFromDimensions(dimensions):
            layers = []
            # add hidden layers
            for (d1, d2) in zip(dimensions, dimensions[1:]):
                layers.append(nn.Linear(d1, d2))
                layers.append(nn.ReLU())
            # swap last activation layer for a sigmoid layer to put output in range from 0 to 1
            layers[-1] = nn.Sigmoid()
            return layers

        # create encoder and decoder sub-networks
        self.encoder = nn.Sequential(*layersFromDimensions(encoderDimensions))  # encoder outputs from 0 to 1
        self.decoder = nn.Sequential(*layersFromDimensions(decoderDimensions))  # decoder outputs from 0 to 1
        # chain sub-networks together
        self.network = nn.Sequential(self.encoder, self.decoder)

    def forward(self, inputs):
        # move input to gpu if necessary
        inputs = inputs.to(device)
        # run the sub-networks on the inputs
        return self.network(inputs)

def trainingLoop(autoencoder, optimizer, loader, loss_function):
    "Performs batch training over a dataset one time."
    dataset_size = len(loader.dataset)
    # loop over each training batch
    for batchNumber, (batchFeatures, batchLabels) in enumerate(loader):
        # calculate the prediction and the loss function output
        prediction = autoencoder(batchFeatures)
        loss = loss_function(prediction, batchLabels)
        # perform backpropagation of errors
        optimizer.zero_grad()   # reset the gradients of the parameters for this batch, since they accumulate by default
        loss.backward()         # backpropagate loss
        optimizer.step()        # update parameters from loss
        # output progress
        if batchNumber % 10 == 0:
            loss = loss.item()
            current = batchNumber * len(batchFeatures)  # number of training examples seen so far
            print(f"Training batch loss: {loss:>9f}  [batch {batchNumber:>3}, starts at image {current:>6d} of {dataset_size:>6d}]")

def testingLoop(autoencoder, loader, loss_function):
    "Performs testing over a dataset one time."
    dataset_size = len(loader.dataset)
    total_loss = 0

    with torch.no_grad():       # don't store derivative information to speed up processing (don't need it since we aren't training)
        for (batchFeatures, batchLabels) in loader:
            prediction = autoencoder(batchFeatures)
            total_loss += loss_function(prediction, batchLabels).item()  # use .item() to access 1-element tensor's element
    average_loss = total_loss / dataset_size
    print(f"Testing average loss: {average_loss:>9f}\n")

# used for development
def callTimed (fn, *args):
    start = time.time()
    result = fn(*args)
    difference = time.time() - start
    print(f"Call took {difference} seconds.")
    return result

def trainAutoencoder(autoencoder, trainingData, testingData, epochs=10):
    # maybe move autoencoder to gpu
    autoencoder = autoencoder.to(device)
    # set training hyperparameters
    batch_size = 30      # higher batch size helps prevent overfitting
    learning_rate = 0.001        # value was determined by testing
    # pick model training method
    loss_function = nn.MSELoss()  # use mean squared error loss function
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, amsgrad=True)  # use the Adam optimizer (adaptive moment estimation) which performs better than stochastic gradient descent when tested
    #optimizer = optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9)  # use stochastic gradient descent to train the network

    # Create dataset loaders, only pinning memory for faster cpu to gpu
    # transfers if the training data is not already stored on the gpu.
    trainingLoader = DataLoader(trainingData, batch_size=batch_size, shuffle=True, pin_memory=not useCudaP)  
    testingLoader = DataLoader(testingData, pin_memory=not useCudaP)
    
    # epochs loop
    for i in range(epochs):
        print(f"Epoch {i}:")
        trainingLoop(autoencoder, optimizer, trainingLoader, loss_function)
        testingLoop(autoencoder, testingLoader, loss_function)
    # notify completion
    print("Complete.")

inDimension = standardWidth * standardHeight  # input image vector size
# bottleneck hidden layer will have the product of these variables as its size
encodedOutputWidth = 15
encodedOutputHeight = 15
middleHiddenLayerSize = encodedOutputWidth * encodedOutputHeight
encoderDimensions = [inDimension, middleHiddenLayerSize]
decoderDimensions = list(reversed(encoderDimensions))

defaultModelName = "autoencoder_model.pt"
def saveAutoencoder(model, path=defaultModelName):
    """Saves information necesarry to restore a trained autoecoder model to a file."""
    torch.save(model.state_dict(), path)
def loadAutoencoder(path=defaultModelName):
    """Returns a new autoencoder with parameters initialized from a file."""
    ae = Autoencoder(encoderDimensions, decoderDimensions)  # make new autoencoder
    ae.to(device)               # maybe move to gpu
    ae.load_state_dict(torch.load(path))  # load trained parameters from file
    ae.eval()                   # necessary to prevent inconsistency compared to pre-serialization
    return ae

# Used in development to test different hyperparameters' performance
def makeTrainAndSaveNewAutoencoder(savePath=defaultModelName):
    """Creates and trains an autoencoder to automatically extract features
     from the dataset for classification. Uses a training and a testing
    set to determine how well it extract features of new images."""

    # Load processed datasets, flattening the images to a 1-d grayscale vector
    trainingSet = ProcessedDataset("train_labels.txt", "./processed_dataset/")
    testingSet = ProcessedDataset("test_labels.txt", "./processed_dataset/")

    # create the autoencoder network
    autoencoder = Autoencoder(encoderDimensions, decoderDimensions)
    # Train the autoencoder
    trainAutoencoder(autoencoder, trainingSet, testingSet)
    # save the autoencoder
    saveAutoencoder(autoencoder, savePath)
    return autoencoder

def encodeImage(autoencoder, imagePath):
    """Takes an autoencoder and image path as input and returns an opencv
    image representing the output of the autoencoder."""
    image = loadImageToTensor(imagePath)
    outputLayer = autoencoder.encoder(image) * 255  # encode image and expand to normal pixel range
    # reshape the output layer to a square image
    outputMatrix = outputLayer.reshape(encodedOutputWidth, encodedOutputHeight)
    cvImage = outputMatrix.to('cpu').detach().numpy()  # detach allows conversion to numpy array
    return cvImage

encodingOutputDirectory = os.path.join(processedDatasetPath, "encoded/")  # the directory to save autoencoder output to
def encodeProcessedImagesForClassification(autoencoder=None):
    """Takes an autoencoder as input, encodes the images listed in train_labels
    and test_labels to a smaller size using the autoencoder, and writes the
    smaller images to files with the same names in the encodingOutputDirectory."""
    # unless an autoencoder was provided, load the default pretrained one
    if autoencoder == None:
        autoencoder = loadAutoencoder()
    def processLabelsFile(labelsFilePath):
        # open the file listing images to encode
        inFile = open(labelsFilePath)
        lines = inFile.read().split('\n')
        inFile.close()
        # encode each image
        for line in lines:
            # skip empty lines
            if line != "":
                # load the image
                imageName, labelString = line.split(',')
                inputPath = os.path.join(processedDatasetPath, imageName)
                outputImage = encodeImage(autoencoder, inputPath)
                # save the encoded image to a file
                outputPath = os.path.join(encodingOutputDirectory, imageName)
                cv.imwrite(outputPath, outputImage)
    # encode all images (training and testing)
    processLabelsFile("./labels.txt")

# only for development purposes, to test autoencoder's accuracy visually
def testAutoencoder(autoencoder):
    """For development purposes. Used to check whether the
    autoencoder output is visually acceptable."""
    imagePath = os.path.join(processedDatasetPath, "Acer palmatum17.png")
    image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
    cv.imshow("Plain", image)
    image = torch.from_numpy(image)  # loads the input image into a pytorch tensor
    # convert image to floats
    image = image.to(dtype=torch.float)
    # convert the image from a matrix into a vector
    image = torch.flatten(image)
    # move image tensor to gpu if cuda is enabled and
    # supported on this computer, else do nothing
    image = image.to(device)

    output_vector = autoencoder(image) * 255
    image_matrix_tensor = output_vector.reshape(standardWidth, standardHeight)
    cv_image = image_matrix_tensor.to('cpu').detach().numpy()
    cv.imshow("Decoder test", cv_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def encodeImageWithPretrainedModel(imagePath, outputPath):
    """Encodes a single image with a pretrained autoencoder model."""
    ae = loadAutoencoder(defaultModelName)
    image = encodeImage(ae, imagePath)
    cv.imwrite(outputPath, image)
