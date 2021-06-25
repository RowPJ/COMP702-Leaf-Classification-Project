from process_dataset import processDataset
from autoencoder import separateDatasets, makeTrainAndSaveNewAutoencoder, encodeProcessedImagesForClassification
from svmClassifier import processAutoencodedImages, trainClassifier, testClassifier, testOverall

# The autoencoder training takes a long time on the cpu, so the default is to use
# the pretrained model. The running time can be accelerated by using a cuda-enabled
# gpu and pytorch release.
def performAllStepsOnDataset(usePretrainedAutoencoder=True, usePretrainedSVM=False):  # default is to use pretrained autoencoder to save time
    """Performs pre-processing, segmentation, feature extraction autoencoder
    training and testing, and svm classifier training and testing. Each phase
    can be configured in its file."""

### preprocessing and segmentation
    # preprocess, segment, and postprocess the leafsnap field images to be classified
    print("Preprocessing, segmenting, and cropping dataset images into the ./processed_dataset/ directory and writing ./labels.txt...")
    processDataset()

    # separate the data into training and testing sets
    print("Separating the dataset images into training and testing sets (with split of 90% training, 10% testing) and writing ./train_labels.txt and ./test_labels.txt...")
    separateDatasets(0.9)

### feature extraction
    if not usePretrainedAutoencoder:
        # train an autoencoder to extract features from leaf images,
        # and save the model
        print("Training an autoencoder neural network for feature extraction on the image set...")
        makeTrainAndSaveNewAutoencoder()
    # use the autoencoder to process the segmented training and testing images
    # into smaller feature space images
    print("Encoding the images using the autoencoder into the ./processed_dataset/encoded/ directory...")
    encodeProcessedImagesForClassification()

### classification
    # convert the autoencoded images to libsvm problem format
    print("Converting the encoded training and testing set images to the libsvm problem format and writing ./libsvm_test_problem, ./libsvm_train_problem, and ./libsvm_all_problem...")
    processAutoencodedImages()
    if not usePretrainedSVM:
        # train the support vector machine on the training images, and save the model
        print("Training the support vector machine on the training images and writing ./trained-model-rbf...")
        trainClassifier()
    # test the support vector machine on the testing images
    print("\nTesting the support vector machine on the testing images...")
    testClassifier()
    # test the support vector machine on all images
    print("\nTesting the support vector machine on all images...")
    testOverall()

def main():
    print("Running this file directly allows you to run all steps of dataset processing, from preprocessing through to classification.")
    autoencoderChoice = input("Do you want to train a new feature extraction auto-encoder or use a pretrained auto-encoder? (0 to re-use the pretrained auto-encoder, 1 to train a new one and overwrite the old one)\n> ")
    classifierChoice = input("Do you want to train a new classifer or use a pretrained classifier? (0 to re-use pretrained classifier, 1 to train a new one and overwrite the old one)\n> ")
    performAllStepsOnDataset(autoencoderChoice!="1", classifierChoice!="1")
    pause = input("Done. Enter something to quit...")

if __name__ == "__main__":
    main()
