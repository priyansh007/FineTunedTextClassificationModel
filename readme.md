# Fine Tuned Text Classification Model

This project focuses on classifying the emotions expressed in tweets using the [Emotion Detection from Text dataset](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text) available on Kaggle. The goal is to develop a model that accurately predicts the emotions associated with different tweets.

# Final product and website code

Final website code can be found at Replit flask project: https://replit.com/@2023-summer-nlp/Proud-Porcupines#bargraph.py 
Website: https://proud-porcupines.2023-summer-nlp.repl.co/index.html

## Setup
To set up the project environment, run the following command:
1. Install the required libraries by running the following command
`!pip install -r requirement.txt`

## Dataset Balancing and Preprocessing

To ensure balanced representation of each emotion class, the dataset is balanced using a combination of oversampling and undersampling techniques. This helps prevent bias towards any particular emotion during training.

The text data undergoes preprocessing steps to improve the model's understanding of the tweets. These preprocessing techniques include lemmatization, stemming, punctuation removal, and stopword removal. These steps reduce noise and standardize the text data for better analysis.

1. Dataset file - dataset.csv
2. Code for preprocessing text - preProcessDataFunctions.py

## Model Architecture and Transfer Learning

For the modeling phase, the project utilizes the DistilBERT-base-uncased tokenizer, a transformer-based model designed for natural language processing tasks. Transfer learning is applied by leveraging the "Seethal/sentiment_analysis_generic_dataset" pretrained model, which has already learned from a large amount of text data.

## Training and Evaluation

The model is trained using the preprocessed data and the pretrained model. The performance of the model is evaluated using a separate test dataset. The achieved accuracy on the test data ranges from 60% to 70%.

3. File for training - trainData.py
4. class for dataset - tweetDataset.py
   
## Emotions Classification

The project aims to predict the following seven emotions:

0: 'anger', 
1: 'love', 
2: 'neutral', 
3: 'positive', 
4: 'sadness', 
5: 'surprise', 
6: 'worry'

The trained model analyzes the content of a given tweet and predicts the most appropriate emotion label based on its content.

## Model Deployment

To ensure easy accessibility and future usage, the trained model is deployed using the Hugging Face model hub. This enables seamless integration and utilization of the model in future applications or projects.

5. File for deployment - deploy.py
6. All notebook files - notebooks folder

## Conclusion

This project provides a solution for classifying emotions expressed in tweets. By leveraging transfer learning, preprocessing techniques, and model deployment, the developed model achieves significant accuracy in predicting the emotions associated with the tweets.
