import numpy as np
import matplotlib.pyplot as plt
from numpy import random
# import random
import time
from collections import deque

class layer:
    k = 0
    n = 0
    number = 0
    z = []
    weights = []
    bias = []

    def __init__(self, n, k, number):
        self.n = n
        self.k = k
        self.number = number
        self.weights = random.normal(size=(self.k, self.n))
        # self.z=np.zeros(shape=(self.k, 1))
        self.bias = np.zeros(shape=(self.k, 1))

    def get_weights(self):
        return self.weights

    def set_weights(self, updated_value):
        self.weights = updated_value

    def get_bias(self):
        return self.bias

    def set_bias(self, updated_value):
        self.bias = updated_value

    def calculate_next_nodes(self, present_a):

        self.z = np.add(np.matmul(self.weights, present_a), self.bias)
        result = sigmoid(self.z)
        return result

    def calculate_gradian_of_cost_W(self, new_a, output, present_a):

        grad_w = np.zeros(shape=(self.k, self.n))

        if self.number == 3:

            grad_w = np.matmul(2 * (new_a - output) * sigmoid_derivative(self.z), present_a.transpose())


        else:

            grad_w = np.matmul((output) * sigmoid_derivative(self.z), present_a.transpose())

        return grad_w

    def calculate_gradian_of_cost_B(self, new_a, output, present_a):
        grad_b = np.zeros(shape=(self.k, 1))

        if self.number == 3:

            grad_b = 2 * (new_a - output) * sigmoid_derivative(self.z)

        else:

            grad_b = output * sigmoid_derivative(self.z)

        return grad_b

    def calculate_gradian_of_cost_present_a(self, new_a, output, present_a):
        grad_a = np.zeros(shape=(self.n, 1))

        if self.number == 3:

            grad_a = np.matmul(np.transpose(self.weights), (2 * (new_a - output) * sigmoid_derivative(self.z)))


        elif self.number == 2:

            grad_a = np.matmul(np.transpose(self.weights), (output * sigmoid_derivative(self.z)))

        return grad_a


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def sigmoid_derivative(x):

    return np.multiply(sigmoid(x), (1 - sigmoid(x)))


def divide_chunks(l, n):

    for i in range(0, len(l), n):
        yield l[i:i + n]


def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


def read_train_set():
    # Reading The Train Set
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)
    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        train_set.append((image, label))


def read_test_set():
    # Reading The Test Set
    test_images_file = open('t10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)
    test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)
    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    # print(num_of_test_images)
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        # print(image.shape)
        for j in range(4):
            image=np.delete(image,1,0)
        # print(image.shape)
        for i in range(4):
            image=np.insert(image,i,0,0)
        # print(image.shape)
        test_set.append((image, label))



def feed_forward(first_layer, second_layer, third_layer):
    accuracy = 0
    for i in range(len(test_set)):

        input = np.zeros(shape=(784, 1))
        input = test_set[i][0]
        second_layer_inputs = first_layer.calculate_next_nodes(input)
        third_layer_inputs = second_layer.calculate_next_nodes(second_layer_inputs)
        fourth_layer_inputs = third_layer.calculate_next_nodes(third_layer_inputs)
        j = 0

        answer = 0
        for row in t_set[i][1]:
            if row == 1:
                answer = j
            j = j + 1


        if np.nanargmax(fourth_layer_inputs) == answer:
            accuracy = accuracy + 1

    print("Accuracy is : ")
    print(accuracy / len(test_set))


def calculate_cost(output, result):
    cost = pow((output - result), 2)
    return sum(cost)


start = time.time()

train_set = []
test_set = []
t_set=[]
read_train_set()
read_test_set()
t_set=train_set
# t_set=test_set
# first step
first_layer = layer(784, 16, 1)
second_layer = layer(16, 16, 2)
third_layer = layer(16, 10, 3)
fourth_layer = layer(10, 0, 4)

# feed_forward(first_layer,second_layer,third_layer)

# part 2
learning_rate = 1
number_of_epochs = 5
batch_size = 50


average_costs = []

for i in range(number_of_epochs):
    print("epoch  ", i)

    random.shuffle(t_set)
    costs = []

    x = list(divide_chunks(t_set, batch_size))

    for batch in x:

        grad_w_layer1 = np.zeros(shape=(16, 784))
        grad_b_layer1 = np.zeros(shape=(16, 1))
        grad_w_layer2 = np.zeros(shape=(16, 16))
        grad_b_layer2 = np.zeros(shape=(16, 1))
        grad_w_layer3 = np.zeros(shape=(10, 16))
        grad_b_layer3 = np.zeros(shape=(10, 1))

        for image in batch:

            input = image[0]

            second_layer_inputs = first_layer.calculate_next_nodes(input)
            third_layer_inputs = second_layer.calculate_next_nodes(second_layer_inputs)
            fourth_layer_inputs = third_layer.calculate_next_nodes(third_layer_inputs)

            costs.append(calculate_cost(image[1], fourth_layer_inputs))

            grad_w_layer3 = np.add(grad_w_layer3, third_layer.calculate_gradian_of_cost_W(fourth_layer_inputs, image[1],
                                                                                          third_layer_inputs))
            grad_b_layer3 = np.add(grad_b_layer3, third_layer.calculate_gradian_of_cost_B(fourth_layer_inputs, image[1],
                                                                                          third_layer_inputs))

            output_grad_a_3 = third_layer.calculate_gradian_of_cost_present_a(fourth_layer_inputs, image[1],
                                                                              third_layer_inputs)
            grad_w_layer2 = np.add(grad_w_layer2,
                                   second_layer.calculate_gradian_of_cost_W(third_layer_inputs, output_grad_a_3,
                                                                            second_layer_inputs))
            grad_b_layer2 = np.add(grad_b_layer2,
                                   second_layer.calculate_gradian_of_cost_B(third_layer_inputs, output_grad_a_3,
                                                                            second_layer_inputs))
            output_grad_a_2 = second_layer.calculate_gradian_of_cost_present_a(third_layer_inputs, output_grad_a_3,
                                                                               second_layer_inputs)

            grad_w_layer1 = np.add(grad_w_layer1,
                                   first_layer.calculate_gradian_of_cost_W(second_layer_inputs, output_grad_a_2, input))
            grad_b_layer1 = np.add(grad_b_layer1,
                                   first_layer.calculate_gradian_of_cost_B(second_layer_inputs, output_grad_a_2, input))


        first_layer.set_weights(
            np.subtract(first_layer.get_weights(), np.multiply(learning_rate, np.divide(grad_w_layer1, batch_size))))
        first_layer.set_bias(
            np.subtract(first_layer.get_bias(), np.multiply(learning_rate, np.divide(grad_b_layer1, batch_size))))

        second_layer.set_weights(
            np.subtract(second_layer.get_weights(), np.multiply(learning_rate, np.divide(grad_w_layer2, batch_size))))
        second_layer.set_bias(
            np.subtract(second_layer.get_bias(), np.multiply(learning_rate, np.divide(grad_b_layer2, batch_size))))

        third_layer.set_weights(
            np.subtract(third_layer.get_weights(), np.multiply(learning_rate, np.divide(grad_w_layer3, batch_size))))
        third_layer.set_bias(
            np.subtract(third_layer.get_bias(), np.multiply(learning_rate, np.divide(grad_b_layer3, batch_size))))


    average_costs.append((sum(costs)) / len(costs))


feed_forward(first_layer, second_layer, third_layer)

end = time.time()
print("Time:", (end - start))

t = np.arange(0, number_of_epochs, step=1)
x_t = average_costs
plt.plot(t, x_t)
plt.show()