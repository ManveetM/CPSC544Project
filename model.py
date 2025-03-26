import pandas as pd
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time

DATAPATH = 'combined_emotion.csv'
PROCPATH = 'processed_data.npz'

def logModel(X, y):
    # Train - test split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=36)

    # Train logistic regression model
    lrm = LogisticRegression(max_iter=1000)
    t1 = time.time()
    lrm.fit(XTrain, yTrain)
    t2 = time.time()
    print(f'Training took {(t2 - t1):.2f} seconds.')

    # Make predictions
    yPred = lrm.predict(XTest)

    # Evaluate model
    accuracy = accuracy_score(yTest, yPred)
    print("Logistic Model")
    print(f"Accuracy is {accuracy:.4f}")
    print(f"Classification report: \n", classification_report(yTest, yPred))

choice = input("Train model (y/n): ")
if choice == 'y':
    # Load labels from csv
    data = pd.read_csv(DATAPATH)
    y = data['emotion'].values

    # Load sparse matrix
    X = load_npz(PROCPATH)

    logModel(X, y)