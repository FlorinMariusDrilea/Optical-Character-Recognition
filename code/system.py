"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import math
import string
import numpy as np
import utils.utils as utils
import scipy.linalg as lin
import scipy.spatial.distance as dst
from scipy import stats
from scipy.stats import mode
from scipy.signal import convolve2d
from difflib import get_close_matches 


def reduce_dimensions(feature_vectors_full, model):
    """Dummy methods that just takes 1st 10 pixels.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """

    # Get PCA components
    getPCA = np.array(model['components'])
    # Get LDA eigenvector matrix
    getLDA = np.array(model['lda'])

    d_mean = np.mean(feature_vectors_full)
    # substract mean
    center = feature_vectors_full - d_mean
    mean = np.dot(center, getPCA)
    # transform samples using LDA
    feature_vectors_reduce = np.dot(mean,getLDA)

    return feature_vectors_reduce


# Functions used for reducing dimensions
def doPCA(feature_vectors, n_dimensions):
    """ Reduce feature vectors to n_dimensions using PCA and return the matrix v
    
    Params:
    feature_vectors - 2d array, each row of the array is a feature vector
    n_dimension - number of dimensions that reduce the feature vectors to n number
    """

    #compute data covariance matrix
    cov_matrix = np.cov(feature_vectors, rowvar = 0)
	
    #computer first n pca axes
    N = cov_matrix.shape[0]
    w, v = lin.eigh(cov_matrix, eigvals=(N - n_dimensions, N - 1))
    v = np.fliplr(v)

    return v


def doLDA(feature_vectors, labels, n_dimensions):
    """ 
    Reference: https://sebastianraschka.com/Articles/2014_python_lda.html
    Reduce feature vectors to n_dimensions with the help of LDA
    
    Params:
    feature_vectors  - 2d array, each row is a feature vector
    labels - labels from training data
    n_dimensions - number of dimensions to reduce the feature vectors
    """

    # get classes
    classes = np.unique(labels)

    # get the mean v. for each class
    means = {}
    for c in classes: 
    	means[c] = np.array(np.mean(feature_vectors[labels == c], axis = 0))

    # overall mean of the data
    overall = np.array(feature_vectors.mean(axis = 0))

    # calculate within-class scatter matrix
    SW = np.zeros((feature_vectors.shape[1], feature_vectors.shape[1]))
    for c in classes:
    	rw = np.subtract(feature_vectors[labels == c].T, np.expand_dims(means[c], axis = 1))
    	SW = np.add(np.dot(rw, rw.T), SW)
   
    # calculate between-class scatter matrix
    SB = np.zeros((feature_vectors.shape[1], feature_vectors.shape[1]))
    for c in means.keys():
    	n = feature_vectors[labels == c, :].shape[0]
    	SB += np.multiply(n, np.outer((means[c] - overall), (means[c] - overall)))
 
    # find eigenvalue, eigenvector pairs for inv(SW).SB
    m = np.dot(np.linalg.pinv(SW), SB)
    eigenvalues, eigenvectors = np.linalg.eig(m)
    eigenlist = [(eigenvalues[i], eigenvectors[:, i])
    # sort eigenvectors
    for i in range(len(eigenvalues))]    
    eigenlist = sorted(eigenlist, key = lambda x: x[0], reverse = True)

    # take the first n_dimensions eigenvectors
    w = np.array([eigenlist[i][1]

    for i in range(n_dimensions)])

    return w.T


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size
    
    #with open('words.txt') as f:
    #dictionary = [word.rstrip() for word in f]
    
    # Subtract mean from all data points
    datamean = np.mean(fvectors_train_full)
    centered = fvectors_train_full - datamean
    
    # Project points onto PCA axes
    fvectors = np.dot(centered, doPCA(fvectors_train_full, 40))

    # Get dictionary of words from text file
    dictionary = use_dictionary('words.txt')
    # Store W matrix from LDA
    model_data['lda'] = doLDA(fvectors, labels_train, 10).tolist()
    # Store PCA components into the model
    model_data['components'] = doPCA(fvectors_train_full, 40).tolist()
    # Create a new field for noise levels
    model_data['noise_levels'] = []
    # Add dictionary of words to the model
    model_data['dict'] = dictionary

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)

    if bbox_size is None:
        bbox_size = get_bounding_box_size(images_test)

    bbox_height, bbox_width = bbox_size
    maximum_noise = 0
    # Calculate noise for this page
    for i, image in enumerate(images_test):
        padded_image = np.ones(bbox_size) * 255
        height, width = image.shape
        width = min(width, bbox_width)
        height = min(height, bbox_height)
        p_img = padded_image[0:height, 0:width]
        img = image[0:height, 0:width]
        p_img = img
        
        noise = get_estimateNoise(padded_image)
        if i == 0:
            maximum_noise = noise
        else:
            if noise > maximum_noise:
                maximum_noise = noise

    noise = model['noise_levels']
    # Add noise level estimates for each page
    noise.append(maximum_noise)
    noise = model['noise_levels']

    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
   
    return fvectors_test_reduced


def get_estimateNoise(img):
    """  
    Reference : https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image  
    
    Get estimated noise on an image

    Params:
    image - image as a 2d array
    """

    h, w = img.shape
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (w - 2) * (h - 2))

    return sigma
    

def classify_page(page, model):
    """Classifier using nearest neighbour

    Params:

    page - matrix, each row should be classified
    model - storing output of training stage
    """
    # models used
    labels_train = np.array(model['labels_train'])
    train = np.array(model['fvectors_train'])
    model_test = np.sqrt(np.sum(page * page, axis=1))
    model_train = np.sqrt(np.sum(train * train, axis=1))
    # get cosine distance
    distance = np.dot(page, train.transpose()) / np.outer(model_test, model_train.transpose())
    # get indices with minimum distances
    near = np.argpartition(distance, -20, axis=1)[:, -20:]
    near = np.sort(near, axis = 1)
    # get most frequent label
    f_label, _ = stats.mode(np.take(labels_train, near), axis=1)
    # reshape it
    f_label = f_label.reshape((page.shape[0], ))
    return f_label

    #2
    #labels_train = np.array(model['labels_train'])
    #fvectors_train = np.array(model['fvectors_train'])
    #noise_levels = model['noise_levels']
    #current_level = noise_levels[0]
    #k = 50
    #noise_levels.pop(0)
    #model['noise_levels'] = noise_levels
    #distances = dst.cdist(fvectors_train, page, 'euclidean')
    #sort = np.argsort(distances, axis = 0)
    #sort = sort[:k, :]
    #values = labels_train[sort]
    #labels = mode(values, axis = 0)[0]

    #return labels[0]



def use_dictionary(file):
    """ get dictionary and return a list of words

    Params:
    file - text file name
    """

    dictionary = []
    for line in file:
    	dictionary.append(line.strip('\n'))

    return dictionary


""" 
FUNCTIONS USED FOR CORRECT ERRORS
"""

def add_punctuation(word1, word2):
    """ Adds punctuation in the second word

    Params:
    word1 - the word from which to take the punctuation and positions
    word2 - the word in which to insert punctuation
    """
    word2 = list(word2)
    for j, length in enumerate(word1):
        if length in string.punctuation: word2.insert(j, length)

    return ''.join(word2)


def remove_punctuation(word):
    """ Removes punctuation

    Params:
    word - word where the punctuation is removed
    """
    word = "".join(w for w in word if w not in string.punctuation)

    return word

def close_match(string, dictionary):
    """Returns the closest match for a word in the given dictionary

    parameters:

    string - string to be replaced
    dictionary - list of strings, the dictionary
    in which to find the closest match
    """
    
    leng = filter(lambda x: len(x) == len(string), dictionary)
    
    # max 1 character has to be changed
    for x, match in enumerate(leng):
        if (string_distance(match, string) == 1):
            return match
    
    # max 2 character has to be changed
    for x, match in enumerate(leng):
        if (len(match) < 4 and string_distance(match, string) <= 2):
            return match
    
    # max 3 character has to be changed
    for x, match in enumerate(leng):
        if (len(match) < 6 and string_distance(match, string) <= 3):
            return match

    return string


def string_distance(string1, string2):
    """Calculate the distance between two strings

    Params:
    string1 - first string
    string2 - second string
    """
    distance = sum([1 for x, y in zip(string1,string2) if x.lower() != y.lower()])
    return distance

def word_correction(word, d):
    """Corrects a string if it has errors

    Parmas:

    word - the word to be corrected
    d - list of strings in dictionary
    """
    
    string = word

    if(not(any(remove_punctuation(word).lower() == wd for wd in d))):
        string = close_match(remove_punctuation(word).lower(), d)
        string = add_punctuation(word, string)
    
    return string



def correct_errors(page, labels, bboxes, model):
    """ Corrects the misclassified labels

    Params:

    page - 2d array, each row is a feature vetcor to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array,each row gives the 4 bouding boox coords of the character
    model - dictionary, stores the output of the training stage
    """

    dictionary = model['dict']
    start = 0

    # differences between boxes
    differences = np.diff(bboxes[:, 0])
    # get boxes width
    width = bboxes[:, 2] - bboxes[:, 0]
    wd = width[:(differences.size)]
    # boxes's space
    spaces = differences[:] - wd
    # get positions at which each word ends
    end = np.where((spaces > 12) | (spaces < -40))
    
    for i in np.nditer(end):
        # get words as an array of characters
        word = labels[start:(i + 1)]
        stw = ''.join(word)
        # separate corrected words in characters and replace labels
        character_word = word_correction(stw, dictionary)
        character_word = np.array(list(character_word))
        labels[start:(i+1)] = character_word
        start = i + 1

    return labels