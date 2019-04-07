# Developing an OCR system in Python

## 1. Objective

- To build and evaluate an optical character recognition system that can process scanned book pages and turn them into text.

## 2. Background

In the lab classes in the second half of the course you will be experimenting with nearest neighbour based classification and dimensionality reduction techniques. In this assignment you will use the experience you have gained in the labs to implement the classification stage of an optical character recognition (OCR) system for processing scanned book pages.

OCR systems typically have two stages. The first stage, document analysis, finds a sequence of bounding boxes around paragraphs, lines, words and then characters on the page. The second stage looks at the content of each character bounding box and performs the classification, i.e., mapping a set of pixel values onto a character code. In this assignment the first stage has been done for you, so you will be concentrating on the character classification step.

The data in this assignment comes from pages of books. The test data has been artificially corrupted, i.e. random offsets have been added to the pixel values to simulate the effect of a poor quality image.

## 3. The task

Your task is to design a classifier that:

1. uses a feature vector containing no more than 10 dimensions;
2. operates robustly even on low quality, `noisy' image data.

## 4. What you are given

You have been given data for training and testing your systems and some code to get you started.

### 4.1. The data

The data is stored in a subfolder named `data` and is split into data for training and data for evaluation. The data comes from pages from novels.  There are 10 pages for training and 6 pages for testing. The testing pages have progressive amounts of noise added to them, i.e.,test page 1 is the best quality and test page 6 is the poorest quality. For each page there are three files.

1. a `png` format image file containing an image of the page. You should be able to view these files in any standard image viewing software.
2. a file ending in the extension `.bb.csv`. This is a comma-separated variable file giving the bounding box coordinates of each successive character on the page. Each line represents the position of a single character.
3. a `label.txt` file giving the correct ASCII label for each character on the page. There is a direct correspondence between the lines in the `.bb.csv` file and the `.label.txt` file.


### 5. Test the code provided

Check that you can run the code provided. Open a terminal in CoCalc. Navigate to the directory containing the assignment code,

`cd com2004_labs/OCR_assignment/code/`

Run the training step,

`python3 train.py`

Then run the evaluation step,

`python3 evaluate.py dev`

The code should print out the percentage of correctly classified characters for each page. The dummy code will produce results in the range 3\% to 5\% correct for each page.

### 6. Processing the training data

The function `process_training_data` in `system.py` processes the training data and returns results in a dictionary called `model_data`. The program `train.py` calls `process_training_data` and saves the resulting `model_data` dictionary to the file `model.json.gz`. This file is then used by the classifier when `evaluate.py` is called. So, any data that your classifier needs must go into this dictionary. For example, if you are using a nearest neighbour classifier then the dictionary must contain the feature vectors and labels for the complete training set. If you are using a parametric classifier then the dictionary must contain the classifier's parameters. The function is currently written with a nearest neighbour classifier in mind. Read it carefully and understand how to adapt it for your chosen approach.

## Feature Extraction

I extracted the image inside each bounding box and converted it into feature vectors and after that is performed the dimensionality reduction.

I added estimated noise to data in order to make it more robust.

I used 2 dimensionality reduction methods, PCA and LDA.
Firstly I started with PCA and I got results for the last page of 24-28%, so I mixed it with LDA to get better results or the noisy pages.
PCA returns the axes where the data is spread the most and LDA returns the axes where data is best separated.

LDA performs better. The scores for clean pages are lower (around 1%) but the scores for the noisy ones are higher with 2-3%.

## Classifier

I implemented the classifier using nearest neighbour method.

The results are good for the noisy pages but for the less noisy ones, 1-2 had worse results.

Nearest neighbours gave slightly worse results for the less noisy pages by a small precentage and gave better results for the noisy ones with (1-3%).
Depending on setting the minimum distances the noisy pages will increase and the clear ones will decrease with 1-2%.

## Error Correction

Error correction is separating the characters into multiple words.

For error correction I am using a dictionary with words where I check words from the pages to the ones from the dictionary and modify it if it's the case, putting the corrected words in characters and replacing the labels. In the end the labels are returned.

For this function i used other functions like: word_correction - used to correct a string with errors matching a word from the text with one from the dictionary with the help of the function of getting a close match and the other 2 function of adding and removing punctuation from the second word compared.

In correct errors we start with the label 0 and keep informations during the for loop, growing the labels with i+1 until the end of it.

The scores after using error correction are lower even though they .

## Performance

The percentage errors (to 1 decimal place) for the development data are
as follows:

The percentage after word correction.

- Page 1: [94.8%]
- Page 2: [94.3%]
- Page 3: [83.6%]
- Page 4: [66.3%]
- Page 5: [51.1%]
- Page 6: [40.5%]

## Other information

To grow the percentage of the noisy pages (make it more robust) with 3-5% I had to get lower percentage on the first (clean) pages with 2-3%.