import pandas as pd
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time

DATAPATH = 'combined_emotion.csv'
PROCPATH = 'processed_data.npz'
LABELPATH = 'labels.npy'

cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=36)

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

def mnBayes(X, y):
    # Train-Test split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=36)

    # Get class weights for balancing
    classWeights = compute_class_weight(
        class_weight='balanced', classes=np.unique(yTrain), y=yTrain
    )

    # Convert to dictionary to assign corresponding weights to samples
    classWeightDict = {clss: weight for clss, weight in zip(np.unique(yTrain), classWeights)}

    # Assign sample weights corresponding to their class
    sampleWeights = np.array([classWeightDict[label] for label in yTrain])

    # Train Multinomial Bayes model
    model = MultinomialNB()
    model.fit(XTrain, yTrain, sample_weight=sampleWeights)

    # Calculate cross validation
    cvScore = cross_val_score(model, XTrain, yTrain, cv=cv, scoring='balanced_accuracy')

    # Make predictions
    yPred = model.predict(XTest)

    # Evaluate model
    accuracy = balanced_accuracy_score(yTest, yPred)
    print("Multinomial Bayes Model")
    print(f"Cross-Validation Accuracy: {cvScore.mean():.4f}")
    print(f"Accuracy is {accuracy:.4f}")
    print(f"Classification report: \n", classification_report(yTest, yPred))

choice = input("Train model (y/n): ")
if choice == 'y':
    # Load labels from csv
    data = pd.read_csv(DATAPATH)
    y = np.load(LABELPATH, allow_pickle=True)

    # Load sparse matrix
    X = load_npz(PROCPATH)

    #logModel(X, y)
    mnBayes(X, y)