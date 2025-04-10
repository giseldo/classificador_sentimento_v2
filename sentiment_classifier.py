import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv("training_data.csv")
texts = data['text'].values
labels = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(texts, labels)

# predictions = model.predict(X_test)
# print("Acurácia:", accuracy_score(y_test, predictions))

joblib.dump(model, "model.pkl")

