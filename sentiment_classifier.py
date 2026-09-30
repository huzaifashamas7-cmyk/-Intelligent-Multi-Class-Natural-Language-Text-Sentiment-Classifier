import nltk

import pandas as pd

import re
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

df = pd.read_csv("dataset.csv")

print("Original Dataset:")
print(df.head())


# ============================================================
# STEP 2: TEXT PREPROCESSING
# ============================================================

# Create stop-word list
stop_words = set(stopwords.words("english"))

# Create lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_map = {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV
    }

    return tag_map.get(tag, wordnet.NOUN)


# Function for text preprocessing
def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Split sentence into individual words
    words = text.split()

    # Remove stop-words and perform lemmatization
    cleaned_words = []

    for word in words:
        if word not in stop_words:
            pos = get_wordnet_pos(word)
            word = lemmatizer.lemmatize(word, pos=pos)
            cleaned_words.append(word)

    # Join the cleaned words back into a sentence
    cleaned_text = " ".join(cleaned_words)

    return cleaned_text

# Apply preprocessing to every sentence
df["clean_text"] = df["text"].apply(preprocess_text)


# Display original and cleaned text
print("\nOriginal and Cleaned Text:")
print(df[["text", "clean_text"]].head(10))


# ============================================================
# STEP 3: SEPARATE TEXT AND SENTIMENT LABELS
# ============================================================

# X_text contains the cleaned sentences
X_text = df["clean_text"]

# y contains the correct sentiment labels
y = df["sentiment"]


# ============================================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================================

# Split the dataset into training and testing data
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))


# ============================================================
# STEP 5: TF-IDF
# ============================================================

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()


# Learn vocabulary from training data
# and convert training text into numbers
X_train = vectorizer.fit_transform(X_train_text)


# Convert testing text into numbers
# using the vocabulary learned from training data
X_test = vectorizer.transform(X_test_text)


print("\nTF-IDF Information:")
print("Training TF-IDF shape:", X_train.shape)
print("Testing TF-IDF shape:", X_test.shape)


# Display some words/features learned by TF-IDF
print("\nFirst 20 TF-IDF Features:")
print(vectorizer.get_feature_names_out()[:20])


# Display first training sentence
print("\nFirst Training Sentence:")
print(X_train_text.iloc[0])


# Display its TF-IDF numerical values
print("\nTF-IDF Values of First Training Sentence:")
print(X_train[0].toarray())

# ============================================================
# STEP 6: TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

# Create the Logistic Regression model
model = LogisticRegression(max_iter=1000)


# Train the model using training data
model.fit(X_train, y_train)


print("\nModel training completed successfully!")

# ============================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================

# Predict sentiment for testing data
y_pred = model.predict(X_test)


print("\nPredicted Sentiments:")
print(y_pred)

# ============================================================
# STEP 8: MODEL EVALUATION
# ============================================================

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)


# Calculate F1-score
f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\nMacro F1-Score:")
print(f1)


# Display detailed classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Create confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Negative", "Neutral", "Positive"]
)

print("\nConfusion Matrix:")
print(cm)


# Display confusion matrix as a heatmap
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"]
)

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.title("Confusion Matrix")

plt.show()

# ============================================================
# STEP 9: INTERACTIVE SENTIMENT PREDICTION
# ============================================================

print("\n========================================")
print("   SENTIMENT CLASSIFIER")
print("========================================")

user_text = input("\nEnter a sentence: ")


# Preprocess the user's sentence
clean_user_text = preprocess_text(user_text)


# Convert the cleaned text into TF-IDF numbers
user_vector = vectorizer.transform([clean_user_text])


# Predict the sentiment
prediction = model.predict(user_vector)


print("\nOriginal Text:")
print(user_text)

print("\nCleaned Text:")
print(clean_user_text)

print("\nPredicted Sentiment:")
print(prediction[0])