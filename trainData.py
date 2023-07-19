import preProcessDataFunctions
import tweetDataset
import pandas as pd
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer,AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from datasets import load_metric
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from huggingface_hub import notebook_login, HfApi

#0: 'anger', 1: 'love', 2: 'neutral', 3: 'positive', 4: 'sadness', 5: 'surprise', 6: 'worry'
dataframe = pd.read_csv('./dataset.csv')

dataframe['sentiment'] = dataframe['sentiment'].replace(category_mapping)
dataframe.dropna()
X_values = dataframe[['content']]
y_values = dataframe['sentiment']

X_balanced, y_balanced = preProcessDataFunctions.balanceDataframe(X_values,
                                                                  y_values)

row_len = X_balanced.shape[0]
for i in range(row_len):
  X_balanced.iloc[i,0] = preProcessDataFunctions.clean_content(X_balanced.iloc[i,0])

max_string_len = 0
for row in range(row_len):
  row_text = X_balanced.iloc[row,0]
  words = word_tokenize(str(row_text))
  if len(words) > max_string_len:
    max_string_len = len(words)

print(f"Max word length of tweet is {max_string_len}")

#Tokenize data

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
input_ids = []
attention_masks = []
y_label = []
for i in range(0,row_len):
  text = X_balanced.iloc[i,0]
  label = y_balanced.iloc[i]
  tokens = tokenizer(str(text), truncation=True, padding="max_length", max_length=max_string_len+5, return_tensors="pt")
  input_ids.append(tokens['input_ids'][0])
  attention_masks.append(tokens['attention_mask'][0])
  y_label.append(label)

tweet_dataset = tweetDataset.TweetDataset(input_ids, attention_masks, y_label)
print(tweet_dataset[0])
#You can try whatever model you want to try from following array
models = ["nlptown/bert-base-multilingual-uncased-sentiment",
          "distilbert-base-uncased",
          "Seethal/sentiment_analysis_generic_dataset",
          "microsoft/deberta-large-mnli"]

model = AutoModelForSequenceClassification.from_pretrained(models[2])
id2label = #To-Do
label2id = #To-Do
num_labels = 7
model.classifier = nn.Sequential(
    nn.Linear(model.config.hidden_size, num_labels)
)
model.config.num_labels = num_labels
model.config.id2label = id2label
model.config.label2id = label2id
model.resize_token_embeddings(len(tokenizer))
train_size = int(0.8 * len(tweet_dataset))
train_data, val_data = random_split(tweet_dataset, [train_size, len(tweet_dataset) - train_size])
learning_rate = 1e-5
num_epochs = 1
batch_size = 1000
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Device using {device}")

def validation_accuracy():
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in val_loader:
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)

            # Count the correct predictions
            correct_predictions += (predicted_labels == batch_labels.squeeze()).sum().item()
            total_predictions += batch_labels.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy

print(f"Training Started")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch_input_ids = batch["input_ids"].to(device)
        batch_attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
        loss = loss_function(outputs, batch_labels.squeeze())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    val_loss = validation_accuracy()
    print(f"Epoch {epoch+1}: Average training Loss = {avg_loss} Average validation accuracy = {val_loss}")
    model.save_pretrained("./model")
print(f"Training Completed")
tokenizer.save_pretrained("./model")
