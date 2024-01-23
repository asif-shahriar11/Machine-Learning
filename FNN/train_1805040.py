# building a Feed-Forward Neural Network
# 4 basic components: Dense layer, ReLU activation, Dropout, Softmax activation

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as scikit
import seaborn as sns
from sklearn.model_selection import train_test_split
import torchvision.datasets as ds
from torchvision import transforms
import pickle

np.random.seed(0)

def adam_optimizer(weights, bias, weights_grad, bias_grad, t, m, v, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    # Update timestep
    t += 1

    # Update biased first moment estimate
    m['weights'] = beta_1 * m.get('weights', np.zeros_like(weights)) + (1 - beta_1) * weights_grad
    m['bias'] = beta_1 * m.get('bias', np.zeros_like(bias)) + (1 - beta_1) * bias_grad

    # Update biased second raw moment estimate
    v['weights'] = beta_2 * v.get('weights', np.zeros_like(weights)) + (1 - beta_2) * np.square(weights_grad)
    v['bias'] = beta_2 * v.get('bias', np.zeros_like(bias)) + (1 - beta_2) * np.square(bias_grad)

    # Compute bias-corrected first moment estimate
    m_hat_weights = m['weights'] / (1 - beta_1 ** t)
    m_hat_bias = m['bias'] / (1 - beta_1 ** t)

    # Compute bias-corrected second raw moment estimate
    v_hat_weights = v['weights'] / (1 - beta_2 ** t)
    v_hat_bias = v['bias'] / (1 - beta_2 ** t)

    # Update parameters
    weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)
    bias -= learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + epsilon)

    return weights, bias, m, v, t


class Layer:
    def __init__(self) -> None:
        self.input = None # input to the layer
        self.output = None # output of the layer

    def forward(self, input: np.ndarray) -> np.ndarray:
        # computes the output Y of a layer for a given input X
        raise NotImplementedError
    
    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        # computes dE/dX for a given dE/dY (and updates parameters if any)
        raise NotImplementedError
    
    def clear(self) -> None:
        # clears the cache (if any)
        pass
    

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, weight_init: str = 'he') -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight_init = weight_init
        self.weights = None
        self.bias = None

        # Adam optimizer parameters
        self.m = {}
        self.v = {}
        self.t = 0

        self.init_weights()

    def init_weights(self) -> None:
        if self.weight_init == 'xav':  # Xavier Glorot initialization
            stddev = np.sqrt(2 / (self.input_size + self.output_size))
        elif self.weight_init == 'he':  # He initialization
            stddev = np.sqrt(2 / self.input_size)
        else:  # Default initialization
            stddev = 0.01
        
        self.weights = np.random.randn(self.input_size, self.output_size) * stddev
        self.bias = np.zeros((1, self.output_size)) + 0.01


    def forward(self, input: np.ndarray) -> np.ndarray:
        # computes the output Y of a layer for a given input X
        #print("dense forward")
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient: np.ndarray, learning_rate: float = 0.1) -> np.ndarray:
        # computes dE/dX for a given dE/dY (and updates parameters if any)
        #print("dense back")
        weights_gradient = np.dot(self.input.T, output_gradient) # dE/dW = dE/dY * dY/dW = dE/dY * X.T
        bias_gradient = output_gradient.mean(axis=0)

        self.weights, self.bias, self.m, self.v, self.t = adam_optimizer(
            self.weights, self.bias, weights_gradient, bias_gradient, 
            self.t, self.m, self.v, learning_rate
        )

        return np.dot(output_gradient, self.weights.T)
    
    def clear(self) -> None:
        self.input = None
        self.output = None
        self.m = {}
        self.v = {}
        self.t = 0

class ReLU:

    def forward(self, input: np.ndarray) -> np.ndarray:
        # computes the output Y of a layer for a given input X
        self.input = input
        #print("relu forward")
        return np.maximum(input, 0)
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # computes dE/dX for a given dE/dY (and updates parameters if any)
        #print("relu back")
        temp = self.input > 0 
        return output_gradient * temp
    
    def clear(self) -> None:
        self.input = None
        self.output = None


    
class Dropout(Layer):
    def __init__(self, rate: float, training: bool = True) -> None:
        super().__init__()
        self.rate = rate
        self.scale = 1.0 / (1.0 - rate)  # Scaling factor
        self.training = training  # By default, set to training mode

    def forward(self, input: np.ndarray) -> np.ndarray:
        if self.training:
            #print("drop train forward")
            self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape)
            return input * self.mask * self.scale
        else:
            #print("drop test forward")
            return input  # During testing, return the input as is

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.training:
            #print("drop train back")
            return output_gradient * self.mask * self.scale
        else:
            #print("drop test back")
            return output_gradient

    def set_mode(self, training: bool) -> None:
        self.training = training

    def clear(self) -> None:
        self.input = None
        self.output = None
        self.mask = None
    
    

class Softmax:
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        # computes the output Y of a layer for a given input X
        #print("softmax forward")
        input -= np.max(input, axis=1, keepdims=True)  # for numerical stability
        clipped_input = np.clip(input, -200, 200)  # You may need to adjust these values
        exps = np.exp(input)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output
    
    def backward(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        #print("softmax back")
        # computes dE/dX for a given dE/dY (and updates parameters if any)
        return self.output - y
    
    def clear(self) -> None:
        self.input = None
        self.output = None
    


class NeuralNetwork:
    def __init__(self) -> None:
        self.layers = []
        self.loss = []
        self.training = True

    def add(self, layer: Layer) -> None:
        # add a layer to the neural network
        self.layers.append(layer)
    
    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()

    def build(self, input_features: int, input_layer_size: int, output_layer_size: int, training: bool = True, dropout_rate = 0.3) -> None:
        # build the neural network
        self.add(Dense(input_features, input_layer_size))
        self.add(ReLU())
        self.add(Dropout(dropout_rate, training))
        # self.add(Dense(input_layer_size, 256))
        # self.add(ReLU())
        # self.add(Dropout(dropout_rate, training))
        self.add(Dense(input_layer_size, output_layer_size))
        self.add(Softmax())

    def set_mode(self, training: bool) -> None:
        """ Set the mode for the network: training or evaluation. """
        self.training = training
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.set_mode(self.training)


    def forward(self, input: np.ndarray) -> np.ndarray:
        # computes the output Y of a neural network for a given input X
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        # predicts the class of each input example in X
        return np.argmax(self.forward(X), axis=1)
    
    def predict_with_loss(self, X: np.ndarray, y: np.ndarray):
        # predicts the class of each input example in X and returns the loss
        y_pred = self.forward(X)
        loss = self.categorical_cross_entropy(y_pred, y)
        return np.argmax(y_pred, axis=1), loss
    
    
    def categorical_cross_entropy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # computes categorical cross entropy loss
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / y_pred.shape[0]

    def categorical_cross_entropy_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # derivative of categorical cross entropy loss
        return y_pred - y_true

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, learning_rate: float = 0.1, batch_size: int = 32, grad_clip: float = 40.0):
        # trains the neural network on a given dataset
        for e in range(epochs + 1):
            error = 0
            for start_idx in range(0, len(X), batch_size):
                x_batch = X[start_idx:start_idx + batch_size]
                y_batch = y[start_idx:start_idx + batch_size]

                # Forward and backward pass for each batch
                batch_error = 0

                output = self.forward(x_batch)
                batch_error += self.categorical_cross_entropy(output, y_batch)
                grad = self.categorical_cross_entropy_derivative(output, y_batch)
                # print min and max of grad
                #print(np.min(grad), np.max(grad))
                for layer in reversed(self.layers):
                    #grad = np.mean(grad, axis=0)
                    grad = np.clip(grad, -grad_clip, grad_clip)
                    if isinstance(layer, Softmax):
                        grad = layer.backward(grad, y_batch)
                    else:   grad = layer.backward(grad, learning_rate)

                batch_error /= len(x_batch)
                error += batch_error

            error /= (len(X) / batch_size)

            if e  == epochs:
                # calculate accuracy and f1 score on training set and validation set
                y_pred = self.predict(X)
                y_test_indices = np.argmax(y, axis=1)
                accuracy = scikit.accuracy_score(y_pred, y_test_indices)
                f1 = scikit.f1_score(y_pred, y_test_indices, average='macro')
                print('Training loss: {:.4f} Accuracy: {:.4f}  F1: {:.4f}'.format(error, accuracy, f1))
            
        return error, accuracy, f1   



        
# helper functions
    
def one_hot_encoding(labels, num_classes):
    labels = labels - 1
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def preprocess(X, y, num_classes):
    # Normalize
    X = X / 255.0
    # Reshape
    X = X.reshape(-1, 28*28)
    # One-hot encode labels
    y_encoded = one_hot_encoding(y, num_classes)

    return X, y_encoded


def train_model(X_train, y_train, X_val, y_val, num_classes=26, epochs=50, learning_rate=5e-4, dropout_rate = 0.1,
                input_layer_size = 128, batch_size=1024):
    # build the neural network model
    net = NeuralNetwork()
    net.build(input_features=28*28, input_layer_size=input_layer_size, output_layer_size=num_classes, training=True, dropout_rate=dropout_rate)
    # train the neural network
    print('Training...')
    t_loss, t_accuracy, t_f1 = net.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
    print('Training completed.')
    # evaluate the model
    net.set_mode(training=False)
    # y_pred = net.predict(X_val)
    y_pred, loss = net.predict_with_loss(X_val, y_val)
    y_val_indices = np.argmax(y_val, axis=1)
    
    accuracy = scikit.accuracy_score(y_pred, y_val_indices)
    f1 = scikit.f1_score(y_pred, y_val_indices, average='macro')
    # print('Validation Loss:', loss)
    # print('Validation Accuracy:', accuracy)
    # print('Validation F1:', f1)
    # plot confusion matrix
    plot_confusion_matrix(y_val_indices, y_pred, f"Learning Rate: {learning_rate}, Input Layer Size: {input_layer_size}")
    return loss, accuracy, f1, t_loss, t_accuracy, t_f1

# Function to plot loss, accuracy, f1 score and confusion matrix
def plot_metric(metric1, metric2, metric_name, title):
    plt.figure(figsize=(12, 4))
    plt.plot(metric1, label='test', color='blue')  # Blue color for test
    plt.plot(metric2, label='train', color='orange')  # Orange color for train
    plt.title(f'{title} - {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = scikit.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(f'{title} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Function to run models with different learning rates and input layer sizes
def run_models(learning_rates, input_layer_sizes, epochs):
    for lr in learning_rates:
        for input_layer_size in input_layer_sizes:
            losses = []
            accuracies = []
            f1s = []
            t_losses = []
            t_accuracies = []
            t_f1s = []
            for epoch in epochs:
                if epoch == 0: print("learning rate: ", lr, "input layer size: ", input_layer_size)
                loss, accuracy, f1, t_loss, t_accuracy, t_f1 = train_model(X_train, y_train, X_val, y_val, num_classes=26, epochs=epoch, learning_rate=lr, dropout_rate = 0.1,
                    input_layer_size = input_layer_size, batch_size=1024)
                losses.append(loss)
                accuracies.append(accuracy)
                f1s.append(f1)
                t_losses.append(t_loss)
                t_accuracies.append(t_accuracy)
                t_f1s.append(t_f1)
                if epoch % 25 == 0 and epoch != 0:  
                    print("epochs:", epoch, "Validation Loss:", loss, "Validation Accuracy:", accuracy, "Validation F1:", f1)
                    print("-------------------------------------------------------")
                    print("-------------------------------------------------------")

            # plot the loss, accuracy, f1 score and confusion matrix on validation set
            # plot loss
            plot_metric(losses, t_losses, "Loss", f"Learning Rate: {lr}, Input Layer Size: {input_layer_size}")
            # plot accuracy
            plot_metric(accuracies, t_accuracies, "Accuracy", f"Learning Rate: {lr}, Input Layer Size: {input_layer_size}")
            # plot f1 score
            plot_metric(f1s, t_f1s, "F1 Score", f"Learning Rate: {lr}, Input Layer Size: {input_layer_size}")

        


# train the best model and save it as pickle
# best model: learning rate = 5e-4, input layer size = 512, epochs = 50
def save_model(X_train, y_train, X_val, y_val):
    net = NeuralNetwork()
    net.build(input_features=28*28, input_layer_size=512, output_layer_size=26, training=True, dropout_rate=0.1)
    # train the neural network
    print('Training...')
    t_loss, t_accuracy, t_f1 = net.train(X_train, y_train, epochs=50, learning_rate=5e-4, batch_size=1024)
    print('Training completed.')
    # evaluate the model
    net.set_mode(training=False)
    y_pred = net.predict(X_val)
    y_val_indices = np.argmax(y_val, axis=1)
    accuracy = scikit.accuracy_score(y_pred, y_val_indices)
    f1 = scikit.f1_score(y_pred, y_val_indices, average='macro')
    print('Validation Accuracy:', accuracy)
    print('Validation F1:', f1)
    # clear the model
    net.clear()
    # save the model as pickle
    with open('model_1805040.pickle', 'wb') as f:
        pickle.dump(net, f)




# main
if __name__ == '__main__':
    # load data
    train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    # Convert to numpy array and get labels
    X_train_val = np.array(train_validation_dataset.data)
    y_train_val = np.array(train_validation_dataset.targets)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)

    X_train, y_train = preprocess(X_train, y_train, num_classes=26)
    X_val, y_val = preprocess(X_val, y_val, num_classes=26)

    learning_rates = [1e-3, 1e-3, 1e-3, 5e-3, 5e-3, 5e-4, 5e-4]
    input_layer_sizes = [64, 128, 256, 256, 512, 256, 512]
    epochs = [100, 100, 100, 25, 25, 75, 50]
    # run_models(learning_rates, input_layer_sizes, epochs)


