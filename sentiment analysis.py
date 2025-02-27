import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Load dataset
file_path = Path.home() / "Downloads" / "sentimentdataset.csv"
train = pd.read_csv(file_path)
train = train[['Text', 'Sentiment']]

# Preprocess text
# train['Text'] = train['Text'].str.replace('&lt;.*?&gt;', ' ', regex=True)

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train['Text'] = train['Text'].apply(clean_text)

# Count instances per sentiment class
class_counts = train['Sentiment'].value_counts()

# Define a threshold for rare classes
threshold = 5
rare_classes = class_counts[class_counts < threshold].index

# Replace rare classes with "Other"
train['Sentiment'] = train['Sentiment'].apply(lambda x: "Other" if x in rare_classes else x)


# Encode labels
label_encoder = LabelEncoder()
train['Sentiment'] = label_encoder.fit_transform(train['Sentiment'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    train['Text'], train['Sentiment'], test_size=0.2, random_state=42, stratify=train['Sentiment']
)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Ensure DataFrame format
total_features = vectorizer.get_feature_names_out()
X_train_df = pd.DataFrame(X_train, columns=total_features)
X_test_df = pd.DataFrame(X_test, columns=total_features)

# Train XGBoost Model
xgmodel = xgboost.XGBClassifier(
    max_depth=24,
    random_state=4430,
    eval_metric="mlogloss",
    learning_rate=0.059,
    n_estimators=500
)

xgmodel.fit(X_train_df, y_train)

# Evaluate Model
y_pred = xgmodel.predict_proba(X_test_df)
auc_score = roc_auc_score(y_test, y_pred, multi_class="ovr")
print(f"Model AUC: {auc_score:.4f}")
