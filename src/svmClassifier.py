import cv2 as cv
from autoencoder import encodingOutputDirectory, encodedOutputWidth, encodedOutputHeight
from libsvm.svmutil import *
import os

# helper for processAutoencodedImages
def libsvmEncodeImage(classNumber, path):
    "Returns a string that represents the image at the given path in sparse-matrix libsvm format."
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    width, height = encodedOutputWidth, encodedOutputHeight
    outputLine = str(classNumber)
    # account for each image pixel in the output line
    for i in range(height):
        for j in range(width):
            # get intensity
            pixel = image[i, j]
            # if intensity is non-zero, save it (because image is encoded sparsely)
            if pixel != 0:
                # calculate a feature number for the pixel, starting from 1
                featureNumber = 1 + (i * width + j)
                outputLine += f" {featureNumber}:{pixel}"
    # end the image's line and return it
    outputLine += "\n"
    return outputLine

def processAutoencodedImages():
    """Creates training and testing libsvm problem files by reading test_labels.txt
    and train_labels.txt. The problem files created are named libsvm_train_problem
    and libsvm_test_problem."""
    def processLabelsFile(labelsFilePath, outputPath):
        # open the file listing image names and labels
        inFile = open(labelsFilePath)
        lines = inFile.read().split('\n')
        inFile.close()
        outFile = open(outputPath, 'w')
        # encode each image into the problem file, with its label
        for line in lines:
            # skip empty lines
            if line != "":
                # load the image
                imageName, labelString = line.split(',')
                label = int(labelString)
                inputPath = os.path.join(encodingOutputDirectory, imageName)
                # convert to a line in the libsvm problem file
                problemText = libsvmEncodeImage(label, inputPath)
                # write it to the file
                outFile.write(problemText)
    # process both training and testing label files
    processLabelsFile("./test_labels.txt", "./libsvm_test_problem")
    processLabelsFile("./train_labels.txt", "./libsvm_train_problem")
    # process all images for final test
    processLabelsFile("./labels.txt", "./libsvm_all_problem")

def trainClassifier():
    """Uses the problem file libsvm_train_problem to train an svm leaf classifier."""
    # load problem
    print("Loading training data...")
    y, x = svm_read_problem("./libsvm_train_problem")
    problem = svm_problem(y, x)
    # train model
    print("Training model...")
    parameter = svm_parameter("-c 1 -g 10")
    model = svm_train(problem, parameter)
    # save model
    print("Saving model...")
    svm_save_model("./trained-model-rbf", model)

def testClassifier():
    """Uses the problem file libsvm_test_problem to test a pre-trained svm leaf classifier."""
    # load problem
    print("Loading testing data...")
    y, x = svm_read_problem("./libsvm_test_problem")
    problem = svm_problem(y, x)

    # load pretrained model
    model = svm_load_model("./trained-model-rbf")

    # test model on training data
    print("Testing model...")
    pLabel, pAcc, pVal = svm_predict(y, x, model)

def testOverall():
    """Uses the problem file libsvm_all_problem to test a pre-trained svm leaf classifier."""
    # load problem
    print("Loading testing data...")
    y, x = svm_read_problem("./libsvm_all_problem")
    problem = svm_problem(y, x)

    # load pretrained model
    model = svm_load_model("./trained-model-rbf")

    # test model on training data
    print("Testing model...")
    pLabel, pAcc, pVal = svm_predict(y, x, model)
