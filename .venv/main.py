import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk

nltk.download('stopwords')

# Function to clean and preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

# Function to count URLs in the urls column
def count_urls(urls):
    if not urls or urls == '[]':
        return 0
    else:
        # Assuming URLs are comma-separated within square brackets
        urls = urls.strip('[]').split(', ')
        return len(urls)

# Read the dataset
df = pd.read_csv('Nazario_5.csv')

# Preprocess text columns
df['subject'] = df['subject'].apply(preprocess_text)
df['body'] = df['body'].apply(preprocess_text)

# Combine subject and body for a full text representation
df['full_text'] = df['subject'] + ' ' + df['body']

# Remove stopwords from the full text
df['full_text'] = df['full_text'].apply(remove_stopwords)

# Count URLs and add as a feature
df['url_count'] = df['urls'].apply(count_urls)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# Fit the vectorizer to the full texts and transform
tfidf_matrix = vectorizer.fit_transform(df['full_text'])

# Create a DataFrame from the TF-IDF matrix
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Add the URL count as a feature
tfidf_df['url_count'] = df['url_count']

# Encode the labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Add the encoded labels to the TF-IDF DataFrame
tfidf_df['label'] = df['label']

# Save the vectorized text with labels to a file
tfidf_df.to_csv('tfidf_vectorized_emails.csv', index=False)

# Save the entire preprocessed dataframe with original text and labels
df.to_csv('PhishingEmail_pre.csv', index=False, encoding='utf-8')

print("Preprocessing complete. Files 'tfidf_vectorized_emails.csv' and 'PhishingEmail_pre.csv' have been saved.")
