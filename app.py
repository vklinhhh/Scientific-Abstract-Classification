import streamlit as st
import joblib
import numpy as np
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load saved models
@st.cache_data(experimental_allow_widgets=True)
def load_models():
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=7)
    device = torch.device("cpu")
    roberta_model.to(device)
    label_encoder = LabelEncoder()
    label_encoder.fit(["CL", "CR", "DC", "DS", "LO", "NI", "SE"])
    lr_combined_model = joblib.load("lr_combined_model.hd5")
    lda_model = joblib.load("lda_model.hd5")
    lr_model = joblib.load("lr_model.hd5")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.hd5")
    return roberta_tokenizer, roberta_model, device, label_encoder, lr_combined_model, lda_model, lr_model, tfidf_vectorizer

# Cache the loaded models using st.cache_data


max_length = 128
batch_size = 16

def tokenize_text(texts, tokenizer, max_length):
    inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return inputs

def majority_vote(predictions_list, label_encoder):
    final_predictions = []
    for i in range(len(predictions_list[0])):
        counts = np.bincount([label_encoder.transform([preds[i]])[0] for preds in predictions_list])
        final_predictions.append(np.argmax(counts))
    return final_predictions

class_mapping = {
    "CL": "Computation and Language",
    "CR": "Cryptography and Security",
    "DC": "Distributed and Cluster Computing",
    "DS": "Data Structures and Algorithms",
    "LO": "Logic in Computer Science",
    "NI": "Networking and Internet Architecture",
    "SE": "Software Engineer"
}
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Define the Streamlit app
def main():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://t4.ftcdn.net/jpg/02/50/48/73/360_F_250487368_Jsjh3z4nvsVwxHh833QEFintTjkjAIeJ.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Scientific Article Abstract Classification App")
    # Add a summary text or description
    st.write("<center><i>Authors: Võ Khánh Linh , Hoàng Nguyễn Phúc, Nguyễn Văn Khánh.<br> \
             This app is a demo for class <b>Phương Pháp Nghiên Cứu Khoa Học</b> at HCMUS - VNU.</i></center>", unsafe_allow_html=True)
    st.write("This app allows you to classify text documents into different classes using an ensemble of models.")
    st.write("You can enter a text document and the app will predict its class using TF-IDF + Logistic Regression, RoBERTa single and RoBERTa + LDA models.")
    st.markdown("**Instructions:**")
    st.write("1. Enter a text document in the text area below.")
    st.write("2. Click the 'Predict' button to get the predicted class label.")
    text_input = st.text_area("Input Article's abstract:", "")

    if st.button("Predict"):
        roberta_tokenizer, roberta_model, device, label_encoder, lr_combined_model, lda_model, lr_model, tfidf_vectorizer = load_models()

        # Preprocess the input text for RoBERTa
        roberta_test_inputs = tokenize_text([text_input], roberta_tokenizer, max_length)
        with torch.no_grad():
            roberta_test_outputs = roberta_model(**roberta_test_inputs.to(device), output_hidden_states=True)[0].cpu().detach().numpy()

        tfidf_input = tfidf_vectorizer.transform([text_input])
        lr_pred = lr_model.predict(tfidf_input)

        # Get LDA topics for the input
        lda_topic = lda_model.transform(tfidf_input)

        # Combine test features for the ensemble model
        combined_features_test = np.concatenate((roberta_test_outputs, lda_topic), axis=1)

        # Make predictions using the ensemble model
        lr_combined_pred = lr_combined_model.predict(combined_features_test)

        # Use the majority_vote function
        final_prediction = majority_vote([lr_pred, lr_combined_pred], label_encoder)

        # Decode the predicted class label
        predicted_label = label_encoder.inverse_transform([final_prediction])[0]
        # Map predicted class abbreviation to full context
        full_label = class_mapping.get(predicted_label, "Unknown")

        st.write("Predicted Class:", full_label)

if __name__ == "__main__":
    main()
