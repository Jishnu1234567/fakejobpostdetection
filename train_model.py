import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('fake_job_postings_balanced.csv')

# Drop rows with missing text data
df = df.dropna(subset=['description', 'fraudulent'])

# Features and target
X = df['description']
y = df['fraudulent']

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train classifier with balanced class weight
clf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(clf, 'ml_model/random_forest_model.pkl')
joblib.dump(vectorizer, 'ml_model/vectorizer.pkl')
