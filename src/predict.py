import pickle

from src.preprocess import preprocess_text

model = pickle.load(open("models/ml_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_email(text):

    clean = preprocess_text(text)

    vector = vectorizer.transform([clean])

    prediction = model.predict(vector)

    return prediction[0]
