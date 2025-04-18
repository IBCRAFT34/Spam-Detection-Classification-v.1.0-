import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

class SpamModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None

    def load_data(self, path='combined_data.csv'):
        data = pd.read_csv (path)
        data = data[['label', 'Message']]
        return data

    def train_model(self, data):
        x = data['Message']
        y = data['label']

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', LogisticRegression(solver='liblinear', random_state=42))
        ])

        self.pipeline.fit(x, y)
        return self.pipeline

    def predict(self, message):
        return self.pipeline.predict([message])[0]

    def predict_proba(self, message):
        return self.pipeline.predict_proba([message])[0]

    def save_model(self, path='spam_model.pkl'):
        joblib.dump(self.pipeline, path)

    def load_model(self, path='spam_model.pkl'):
        self.pipeline = joblib.load(path)

if __name__ == "__main__":
    model = SpamModel()
    data = model.load_data('combined_data.csv')
    model.train_model(data)
    model.save_model('spam_model.pkl')
    print("Model berhasil dilatih dan disimpan >w<..!!!")