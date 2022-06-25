import numpy as np  # to import numpy for generating data
import matplotlib.pyplot as plt  # to import matplotlib for plotting data
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import sklearn.metrics as mt
import seaborn as sns
from sklearn.utils import shuffle

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1.0 - z)

# Neural Network class

class NeuralNetwork:
    def __init__(self, inSize, sl2, clsSize, lrt):
        # Constructor expects:\n,  inSize- input size, number of features\n"
        # sl2 - number of neurons in the hidden layer\n",
        # clsSize - number of classes, equals number of neurons in output layer\n",
        # lrt - learning rate\n",

        self.iSz = inSize  # number of input units
        self.oSz = clsSize  # number of output units
        self.hSz = sl2  # number of hidden units

        # Initial assignment of weights \n",
        np.random.seed(42)  ## assigning seed so it generates the same random number all the time. Just to fix the result.\n",
        self.weights1 = (np.random.rand(self.hSz, self.iSz + 1) - 0.5) / np.sqrt(self.iSz)
        self.weights2 = (np.random.rand(self.oSz, self.hSz + 1) - 0.5) / np.sqrt(self.hSz)

        self.output = np.zeros(clsSize)
        self.layer1 = np.zeros(self.hSz)
        self.eta = lrt


    def feedforward(self, x):

        x_bias = np.r_[1, x]
        a1 = np.dot(self.weights1, x_bias)
        self.layer1 = sigmoid(a1)
        layer1_c = np.r_[1, self.layer1]
        a2 = np.dot(self.weights2, layer1_c)
        self.output = sigmoid(a2)


    def backprop(self, x, trg):

        sigma_3 = (trg - self.output)  # outer layer error
        sigma_3 = np.reshape(sigma_3, (self.oSz, 1))

        layer1_c = np.r_[1, self.layer1]  # hidden layer activations+bias
        sigma_2 = np.dot(self.weights2.T, sigma_3)

        deactive = sigmoid_derivative(layer1_c)  # deactive
        deactive = np.reshape(deactive, (self.hSz + 1, 1))

        sigma_2 = np.multiply(sigma_2, deactive)  # hidden layer error

        x_bias = np.r_[1, x]  # input layer +bias

        delta2 = np.multiply(sigma_3, layer1_c)  # weights2 update
        delta1 = np.multiply(sigma_2[1:], x_bias)

        return delta1, delta2


    def fit(self, X, y, iterNo):

        m = np.shape(X)[0]

        for i in range(iterNo):
            D1 = np.zeros(np.shape(self.weights1))
            D2 = np.zeros(np.shape(self.weights2))
            # new_error = 0",
            for j in range(m):
                self.feedforward(X[j])
                yt = np.zeros(self.oSz)
                yt[int(
                    y[j])] = 1  # the output is converted to vector, so if class of a sample is 1, then yt=[0 1 0]\n",
                [delta1, delta2] = self.backprop(X[j], yt)
                D1 = D1 + delta1
                D2 = D2 + delta2

            self.weights1 = self.weights1 + self.eta * (D1 / m)  # weights1 are updated only ones after one epoch\n",
            self.weights2 = self.weights2 + self.eta * (D2 / m)  # weights2 are updated only ones after one epoch\n",

    # This function is called for prediction\n",

    def predict(self, X):
        m = np.shape(X)[0]
        y_proba = np.zeros((m, 3))
        y = np.zeros(m)
        for i in range(m):
            self.feedforward(X[i])
            y_proba[i, :] = self.output  # the outputs of the network are the probabilities\n",
            y[i] = np.argmax(self.output)  # here we convert the probabilities to classes\n",
        return y, y_proba


iris = datasets.load_iris()
x = iris.data
y = iris.target

'''
NN = NeuralNetwork(4, 2, 3, 0.2)
NN.fit(x, y, 1000)
y1, y2 = NN.predict(x)
print(y1)
print(y2)
'''
x, y = shuffle(x, y) # I shuffled data first

# I splitted data into three part as train, validation and test set:
train_x = x[0:100, :]
train_y = y[0:100]

validation_x = x[100:125, :]
validation_y = y[100:125]

test_x = x[125:, :]
test_y = y[125:]

lrt = [0.1, 0.2, 0.3]
sl2 = [2, 3, 4]
iterNo = [1000, 500]
inSize = 4
clsSize = 3

def accuracy_calculator(predicted_set, target_set):
    number = 0
    for a in range(len(predicted_set)):
        if(predicted_set[a] == target_set[a]):
            number = number + 1

    accuracy = number / predicted_set.size

    return number, accuracy

NN = []
accurancies = []

for lr in lrt:
    nn_lr = NeuralNetwork(4, sl2[0], 3, lr)
    nn_lr.fit(train_x, train_y, iterNo[0])
    NN.append(nn_lr)

a = 0
for nn_lr in NN:
    y_predict, y_proba = nn_lr.predict(validation_x)
    correct_prediction, accuracy = accuracy_calculator(y_predict, validation_y)
    print("Correct Prediction :", correct_prediction)
    print("Accuracy when lrt is: " + str(lrt[a]) + " --> ", accuracy)
    accurancies.append(accuracy)
    #print("Accuruancies:", accurancies)
    temp_largest_accuracy = max(accurancies)
    #print("temp_largest_accuracy:", temp_largest_accuracy)
    max_index = accurancies.index(temp_largest_accuracy)
    #print("max index:", max_index)
    a = a + 1

print("Accurancies list: ", accurancies)
#print(max_index)
print("BEST LEARNING RATE IS: ", lrt[max_index])
y_predict, y_proba = NN[max_index].predict(test_x)
print("Best model on test set:", accuracy_calculator(y_predict, test_y))


accurancies.clear()
NN.clear()

print("\n PART b: ")

for SL2 in sl2:
    nn_lr1 = NeuralNetwork(4, SL2, 3, lrt[1])
    nn_lr1.fit(train_x, train_y, iterNo[1])
    NN.append(nn_lr1)

i = 0
for nn_lr1 in NN:
    y_predict, y_proba = nn_lr1.predict(validation_x)
    correct_prediction, accuracy = accuracy_calculator(y_predict, validation_y)
    print("Correct Prediction :", correct_prediction)
    print("Accuracy when sl2 is: " + str(sl2[i]) + " --> ", accuracy)
    accurancies.append(accuracy)
    #print("Accuruancies:", accurancies)
    temp_largest_accuracy = max(accurancies)
    #print("temp_largest_accuracy:", temp_largest_accuracy)
    max_index1 = accurancies.index(temp_largest_accuracy)
    #print("max index", max_index1)
    i = i+1


print("Accurancies list: ", accurancies)
print("BEST NUMBER OF UNITS IN THE HIDDEN LAYER IS: ", sl2[max_index1])
y_predict, y_proba = NN[max_index1].predict(test_x)
print("Best model on test set: ", accuracy_calculator(y_predict, test_y))

###########################################################################################

print("\n PART c: ")

# After completing a and b parts we can decide on which hyperparameters are more proper via combining train set and validation set:

#best_nn = NeuralNetwork(4, 3, 3, 0.2) This is best case that I choose

best_nn = NeuralNetwork(4, sl2[max_index1], 3, lrt[max_index]) # It creates best nn with best hyperparameters
best_nn.fit(train_x, train_y, 500)
y_predict1, y_proba1 = best_nn.predict(validation_x)
correct_prediction1, accuracy1 = accuracy_calculator(y_predict1, validation_y)
print("Accuracy with validation set: ", accuracy1)
y_predict1, y_proba1 = best_nn.predict(test_x)
print("Best model with best hyperparameters on test set:", accuracy_calculator(y_predict1, test_y))


# Creating confusion matrix with test data:
data = confusion_matrix(test_y, y_predict1)
ax = plt.subplot()
sns.heatmap(data, annot=True, fmt='g', ax=ax)

ax.set_xticklabels(iris.target_names)
ax.set_yticklabels(iris.target_names)
ax.set(ylabel="Predicted Label", xlabel="True Label")
plt.title('Confusion Matrix')
plt.show()

