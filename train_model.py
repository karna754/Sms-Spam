import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Load dataset and print columns to inspect
df = pd.read_csv('cleaned_messages2.csv')
print("Columns in the dataset:", df.columns.tolist())
print("First few rows:")
print(df.head())

# Using the actual column names from the CSV
text_column = 'cleaned_message'  # Using the pre-cleaned messages
label_column = 'label'           # The label (spam/ham)

# Remove rows with missing values in the text column
df = df.dropna(subset=[text_column])

# Preprocessing
X = df[text_column]
y = df[label_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'spam_detection_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
print("\nModel and vectorizer saved successfully!")