# OCR assignment report

## Feature Extraction (Max 200 Words)

I extracted the image inside each bounding box and converted it into feature vectors and after that is performed the dimensionality reduction.

I added estimated noise to data in order to make it more robust.

I used 2 dimensionality reduction methods, PCA and LDA.
Firstly I started with PCA and I got results for the last page of 24-28%, so I mixed it with LDA to get better results or the noisy pages.
PCA returns the axes where the data is spread the most and LDA returns the axes where data is best separated.

LDA performs better. The scores for clean pages are lower (around 1%) but the scores for the noisy ones are higher with 2-3%.

## Classifier (Max 200 Words)

I implemented the classifier using nearest neighbour method.

The results are good for the noisy pages but for the less noisy ones, 1-2 had worse results.

Nearest neighbours gave slightly worse results for the less noisy pages by a small precentage and gave better results for the noisy ones with (1-3%).
Depending on setting the minimum distances the noisy pages will increase and the clear ones will decrease with 1-2%.

## Error Correction (Max 200 Words)

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

## Other information (Optional, Max 100 words)

To grow the percentage of the noisy pages (make it more robust) with 3-5% I had to get lower percentage on the first (clean) pages with 2-3%.