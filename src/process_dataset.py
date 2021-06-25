import cv2 as cv
import numpy as np

# Must call processDataset after changing these values.
# Images will be saved as squares.
standardWidth = 50
standardHeight = 50

def loadImage(path, debug=False):
    """Loads the image at the given path in full colour."""
    # load image in colour. ignore alpha channel
    colour_image = cv.imread(path, cv.IMREAD_COLOR)
    if debug:
        cv.imshow("colour", colour_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    # return the colour image
    return colour_image

def segmentImage(image, debug=False):
    """Perform automatic global threshold segmentation based
    on image saturation, extract the region of interest as
    a sub-image, and normalize its dimensions."""
    # convert to hsv colour space for segmentation
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # copy saturation channel to a grayscale image
    saturationChannel = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.mixChannels([hsv_image], [saturationChannel], (1, 0))

    if debug:
        cv.imshow("value channel", valueChannel)
        cv.imshow("saturation channel", saturationChannel)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    # segment the image with automatic threshold selection
    thresholdValue, thresholdSegmented = cv.threshold(saturationChannel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # perform median filtering to remove thresholding noise
    segmentationMask = cv.medianBlur(thresholdSegmented, 5)
    if debug:
        cv.imshow("segmented", segmentationMask)
    
    # extract the region of interest using the segmented image
    # mask to bound the part of the image to extract
    pointsOfInterest = cv.findNonZero(segmentationMask)
    x, y, w, h = cv.boundingRect(pointsOfInterest)
    # black out the image pixels that aren't within the segmentation mask (since they are irrelevant)
    clampedMask = np.clip(segmentationMask, 0, 1)
    # multiply corresponding entries (0 in the segmentation mask will destroy
    # unnecessary pixels, 1 will leave them unchanged)
    blackened = np.multiply(cv.cvtColor(image, cv.COLOR_BGR2GRAY), clampedMask)
    if debug:
        cv.imshow("with segmentation mask", blackened)

    # extract the subimage of interest from the points of the the region of interest
    subimageOfInterest = blackened[y:y+h, x:x+w]
    if debug:
        cv.imshow("subimage", subimageOfInterest)

    # normalize the dimensions of the extracted sub-image
    normalized = cv.resize(subimageOfInterest, (standardWidth, standardHeight))
    if debug:
        cv.imshow("size normalized", normalized)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return normalized

# define constants for processing the leafsnap dataset
leafsnapPath = "../leafsnap-dataset/"
processedDatasetPath = "./processed_dataset/"
# this list is the list of species to classify between. it can be freely changed, after which the system must be re-trained
species = ["Quercus muehlenbergii",
           "Aesculus pavi",
           "Ulmus pumila",
           "Styrax japonica",
           "Acer palmatum",
           "Cryptomeria japonica",
           "Carya glabra",
           "Catalpa speciosa",
           "Salix nigra",
           "Tilia cordata",
           "Eucommia ulmoides",
           "Zelkova serrata",
           "Koelreuteria paniculata",
           "Juniperus virginiana"]
labels = {plant:index for (index,plant) in enumerate(species)}

def processImage(path, outputPath):
    """Loads a dataset image and saves its processed form to a file."""
    image = loadImage(path)
    output = segmentImage(image)
    output = cv.equalizeHist(output)  # equalize the histogram to improve contrast
    cv.imwrite(outputPath, output)
    return output

def processDataset():
    # open the file describing the image locations
    sourcesList = open(leafsnapPath + "leafsnap-dataset-images.txt")
    lines = sourcesList.read().split('\n')[1:]  # ignore first line (it's just headings)
    sourcesList.close()

    # write label for each image to a label file
    labelFile = open("labels.txt", 'w')

    # maps plant types to number of field images read so far
    counts = {name:0 for name in species}
    
    # process each line
    for line in lines:
        # ignore empty newline at the end of the document, if there is one
        if line != "":
            (file_id, image_path, segmented_path, plant, source) = line.split('\t')
            # only use field images
            if plant in species and source == "field":
                # prepare input image path
                inputImagePath = leafsnapPath + image_path
                # prepare output image path
                outputFileName = plant + str(counts[plant]) + ".png"
                outputPath = processedDatasetPath + outputFileName
                counts[plant] += 1
                # process the image with the paths
                output = processImage(inputImagePath, outputPath)
                # output the label for the image
                labelFile.write(outputFileName + ", " + str(labels[plant]) + '\n')
    labelFile.close()
