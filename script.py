import os
import sys
import pandas as pd
import shutil
import tensorflow as tf
import joblib
from pathlib import Path
import csv
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PyPDF2 import PdfReader
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords and other necessary data from nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Convert PDF to text
def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)  
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize the text (split into words)
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatization 
    words = [lemmatizer.lemmatize(word) for word in words]
  
    preprocessed_text = ' '.join(words)
    
    return preprocessed_text


# Load the pre-trained deep learning model, tokenizer, and label map
def load_model_and_assets(model_path, tokenizer_path, label_map_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = joblib.load(f)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading the tokenizer: {e}")
        sys.exit(1)

    try:
        with open(label_map_path, 'rb') as f:
            label_map = joblib.load(f)
        print("Label map loaded successfully.")
    except Exception as e:
        print(f"Error loading the label map: {e}")
        sys.exit(1)

    return model, tokenizer, label_map


# Process the resumes and categorize them
def process_resume(directory_path, model, tokenizer, label_map, max_len):
    results = []

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            print("root : ", root)
            print("files : ", files)
            if filename.endswith('.pdf'):
                file_path = os.path.join(root, filename)
                print(f"Processing file: {filename}")

                try:
                    # Convert the PDF to text
                    resume_text = pdf_to_text(file_path)
                    
                    # Preprocess and tokenize the text
                    processed_text = preprocess_text(resume_text)
                    seq = tokenizer.texts_to_sequences([processed_text])
                    padded_seq = pad_sequences(seq, maxlen=max_len)

                    # Predict the category
                    predictions = model.predict(padded_seq)
                    predicted_category_idx = predictions.argmax(axis=-1)[0]
                    
                    # Get the category label
                    predicted_category = list(label_map.keys())[list(label_map.values()).index(predicted_category_idx)]

                    output_path = os.path.join(directory_path, "outputs")
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    
                    # Move the resume to the corresponding category folder
                    category_path = os.path.join(output_path, predicted_category)
                    if not os.path.exists(category_path):
                        os.makedirs(category_path)
                    shutil.copy(file_path, os.path.join(category_path, filename))
                    
                    # Append the result to the list
                    results.append([filename, predicted_category])
                    print(f"Appended result for {filename}: {predicted_category}")

                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
        
    print(f"Finished processing resumes. Total results: {len(results)}")
    return results
        

# Save the results to a CSV file
def save_results_to_csv(results, output_csv_path):
    try:
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Category'])
            writer.writerows(results)
        print(f"Results saved to {output_csv_path}.")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

# Main function
def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_directory>")  ## The resumes are taken as like the given dataset folders
        sys.exit(1)

    directory_path = sys.argv[1]
    
    # Load the trained model, tokenizer, and label map
    model_path = 'resume_classification_model.h5' 
    tokenizer_path = 'tokenizer.pkl' 
    label_map_path = 'label_map.pkl'
    model, tokenizer, label_map = load_model_and_assets(model_path, tokenizer_path, label_map_path)

    # Process the resumes
    max_len = 200  
    results = process_resume(directory_path, model, tokenizer, label_map, max_len)
    
    # Save the categorization results to a CSV file
    output_csv_path = os.path.join(directory_path, 'categorized_resumes.csv')
    save_results_to_csv(results, output_csv_path)

    print(f"Categorization complete. Results saved to {output_csv_path}.")

if __name__ == "__main__":
    main()
