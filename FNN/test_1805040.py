import pickle
import sklearn.metrics as scikit
import torchvision.datasets as ds
from torchvision import transforms
import numpy as np
import train_1805040 as tr
from train_1805040 import NeuralNetwork, Dense, ReLU, Dropout, Softmax

np.random.seed(0)

independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                             train=False,
                             transform=transforms.ToTensor())

# with open('ids7.pickle', 'rb') as ids7:
#   independent_test_dataset = pickle.load(ids7)


# Convert to numpy array and get labels
X_test = np.array(independent_test_dataset.data)
y_test = np.array(independent_test_dataset.targets)


# preprocess data
X_test, y_test = tr.preprocess(X_test, y_test, 26)


# load pickle file
filename = 'model_1805040.pickle'
with open(filename, 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully")

# predict
y_pred, loss = model.predict_with_loss(X_test, y_test)
y_test_indices = np.argmax(y_test, axis=1)

# print accuracy, f1 score and loss
accuracy = scikit.accuracy_score(y_test_indices, y_pred)
f1_score = scikit.f1_score(y_test_indices, y_pred, average='macro')
print("Loss: ", loss)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1_score)

# plot confusion matrix 
tr.plot_confusion_matrix(y_test_indices, y_pred, title='Confusion matrix for Test Set')







