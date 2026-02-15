Smart Email Priority Classifier

Overview

The Smart Email Priority Classifier is a Machine Learning project that automatically categorizes incoming emails based on their content.
The system uses Natural Language Processing (NLP) techniques to analyze email text and classify it into categories such as spam or normal messages.

This project demonstrates a complete machine learning pipeline, including:

- Data cleaning
- Text preprocessing
- Feature extraction
- Model training
- Model evaluation
- Prediction using a trained model

The goal of this project is to reduce email overload by automatically identifying the type of incoming email.



Features

- Email text preprocessing using NLP
- Automatic email classification
- Machine Learning model using Logistic Regression
- Deep Learning model using PyTorch Neural Network
- Command-line interface for testing predictions
- Modular code structure for easy expansion

---

Tech Stack

- Python
- NumPy
- Pandas
- scikit-learn
- NLTK
- PyTorch

---

Dataset

This project uses the SMS Spam Collection Dataset, which contains thousands of labeled messages classified as spam or normal.

Dataset fields:

- "text" – message content
- "label" – email category

---

Machine Learning Pipeline

1. Load dataset
2. Clean and preprocess text data
3. Tokenization and stopword removal using NLTK
4. Convert text into numerical features using TF-IDF
5. Train classification models
6. Evaluate model accuracy
7. Use trained model for prediction

---

Project Structure

smart-email-priority-classifier
│
├── data
│   └── emails.csv
│
├── models
│   ├── ml_model.pkl
│   └── dl_model.pth
│
├── src
│   ├── preprocess.py
│   ├── train_ml.py
│   ├── train_dl.py
│   └── predict.py
│
├── main.py
├── requirements.txt
└── README.md

---

Installation

Clone the repository:

git clone https://github.com/your-username/smart-email-priority-classifier.git

Move into the project directory:

cd smart-email-priority-classifier

Install dependencies:

pip install -r requirements.txt

---

Training the Model

Train the machine learning model:

python src/train_ml.py

Train the deep learning model:

python src/train_dl.py

---

Running the Classifier

Run the application:

python main.py

Example:

Enter email text: You won a free prize
Email Category: spam

---

Future Improvements

- Add more email categories
- Improve model accuracy
- Build a web interface for real-time email classification
- Deploy the model as an API

---

Author

Aditya Vyas

Machine Learning enthusiast focused on building practical AI applications and learning core ML concepts through hands-on projects.