import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.metrics import classification_report
import joblib

# Load your dataframe
data = pd.read_csv('/Users/mac/Documents/Personal Material/cv/project/scientific-abstract-classification/Data/train.csv', names=["document", "class"], header=0)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data_cl = train_data[train_data["class"]=='CL'].iloc[:100].reset_index()
train_data_cr = train_data[train_data["class"]=='CR'].iloc[:100].reset_index()
train_data_dc = train_data[train_data["class"]=='DC'].iloc[:100].reset_index()
train_data_ds = train_data[train_data["class"]=='DS'].iloc[:100].reset_index()
train_data_lo = train_data[train_data["class"]=='LO'].iloc[:100].reset_index()
train_data_ni = train_data[train_data["class"]=='NI'].iloc[:100].reset_index()
train_data_se = train_data[train_data["class"]=='SE'].iloc[:100].reset_index()
train_data = pd.concat([train_data_cl,train_data_cr,train_data_dc,train_data_ds,train_data_lo,train_data_ni,train_data_se],ignore_index=True).drop(["index"], axis="columns")


# Preprocessing for TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_train = tfidf_vectorizer.fit_transform(train_data['document'])
tfidf_test = tfidf_vectorizer.transform(test_data['document'])

# Train a Logistic Regression model on TF-IDF
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(tfidf_train, train_data['class'])
lr_predictions = lr_model.predict(tfidf_test)
# Save the trained LR model
joblib.dump(lr_model, "lr_model.hd5")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.hd5")

# Load and tokenize the RoBERTa model
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=7)
device = torch.device("cpu")
roberta_model.to(device)

# Save RoBERTa tokenizer and model
roberta_tokenizer.save_pretrained("roberta_tokenizer.hd5")
roberta_model.save_pretrained("roberta_model.hd5")

# Preprocessing for RoBERTa
def tokenize_text(texts, tokenizer, max_length):
    inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return inputs

max_length = 128
batch_size = 16

# Preprocess training data for RoBERTa
roberta_train_inputs = tokenize_text(train_data['document'].tolist(), roberta_tokenizer, max_length)
with torch.no_grad():
    roberta_train_outputs = roberta_model(**roberta_train_inputs.to(device), output_hidden_states= True)[0].cpu().detach().numpy()


# Load LDA model
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)  # Adjust the number of components as needed
lda_topics_train = lda_model.fit_transform(tfidf_train)

# Combine RoBERTa embeddings with LDA topics
combined_features_train = np.concatenate((roberta_train_outputs, lda_topics_train), axis=1)
# Train a Logistic Regression model on combined features
lr_combined_model = LogisticRegression(max_iter=1000)
lr_combined_model.fit(combined_features_train, train_data['class'])
# Save the trained LDA model
joblib.dump(lda_model, "lda_model.hd5")
joblib.dump(lr_combined_model, "lr_combined_model.hd5")