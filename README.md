# Hinglish Sentiment Classifier

This is a simple machine learning project that tries to classify messages into different sentiments.

The model takes a text message (English + Hinglish mix) and predicts whether it is:
- positive
- negative
- sarcastic


## What this project does

People often write messages in Hinglish and sometimes use sarcasm.  
It becomes difficult to understand the actual intent.

This project builds a basic model to detect that using machine learning.


## Approach

- Collected a dataset of mixed Hinglish and English messages  
- Cleaned the data (removed empty rows, fixed format issues)  
- Converted text into numerical form using CountVectorizer  
- Trained a Naive Bayes classifier  
- Tested the model and checked accuracy  


## Tech used

- Python  
- pandas  
- scikit-learn  


## How to run

1. Make sure Python is installed  
2. Install required libraries:
pip install pandas scikit-learn

3. Run the file:python sentiment_model.py

## Sample usage

Run the script and type any message:hello bhai kya scene hai


The model will output the predicted sentiment.


## Notes

- The dataset contains noisy real-world text, so preprocessing was required  
- Some inconsistencies in data were handled during loading  


## What I learned

- How to clean messy data  
- How to convert text into features  
- How to train a basic ML model  
- How to debug errors during implementation  