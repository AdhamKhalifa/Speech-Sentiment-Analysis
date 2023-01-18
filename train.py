# To Train the model, we use a dataset of annotated audio files and use a nueral network to train the model
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from utils import load_data
from utils import load_validation_data

import os
import pickle

# load RAVDESS dataset
X_train, X_test, y_train, y_test = load_data(test_size=0.25)

# print some details
# number of samples in training data
print("Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("Number of features:", X_train.shape[1])
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
model = MLPClassifier(**model_params)

# train the model
print("Training the model...")
model.fit(X_train, y_train)

# predict 25% of data to measure how good the model is
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
# create a log txt file and save the "training accuracy" in it
if not os.path.isdir("results"):
    os.mkdir("results")
with open("results/log.txt", "a") as f:
    f.write("Training accuracy: {}\n".format(accuracy))

# validation of the model
X_valid, y_valid = load_validation_data()
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_true=y_valid, y_pred=y_pred)
print("Validation Accuracy: {:.2f}%".format(accuracy*100))
# write the validation accuracy in the log file
with open("results/log.txt", "a") as f:
    f.write("Validation accuracy: {}\n".format(accuracy))


# now we save the model
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/speech_sentiment.model", "wb"))