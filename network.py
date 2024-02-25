import numpy as np



class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def ileri(self, inp):
        pass

    def geri(self, o, lr):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) # len is output_size and len X[0] is input_size
        self.bias = np.random.randn(output_size, 1) # it is default
        # self.bias = np.random.randn(1, output_size)

    def ileri(self, inp):
        self.input = inp
        return np.dot(self.weights, self.input) + self.bias # .T[0]

    def geri(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, self.input.T)
        self.weights += learning_rate * weight_gradient
        self.bias += learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)


class Act(Layer):
    def __init__(self, act, prime):
        self.act = act
        self.prime = prime

    def ileri(self, inp):
        self.input = inp
        return self.act(self.input)

    def geri(self, o, lr):
        return np.multiply(o, self.act(self.input))

class Lower(Act):
    def __init__(self, val):
        l = lambda x: x * val
        lprime = lambda x: val
        super().__init__(l, lprime)

class Tanh(Act):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


def square(a, b):
    a = np.array([a]).T
    return np.mean(np.power(a - b, 2))

def square_prime(a, b):
    a = np.array([a]).T
    return 2 * (a * b) / np.size(a)

def test():
    X = np.reshape([[0,0],[0,1], [1,0], [1,1]], (4,2,1))
    Y = np.reshape([[0],[1], [1], [0]], (4,1,1))
    network = [
        Dense(2, 10),
        Lower(0.1),
        Dense(10, 5),
        Lower(0.1),
        Dense(5, 1),
        Lower(0.1)
    ]

    epoch = 5000
    lr = 0.01

    for e in range(epoch):
        err = 0
        for x, y in zip(X, Y):
            output = x
            for layer in network:
                output = layer.ileri(output)
                print(output)
                print("----------------------------")

            err += square(y, output)
        
            grad = square_prime(y, output)

            for layer in reversed(network):
                grad = layer.geri(grad, lr)

        err /= len(X)
        print(f"[+] {e + 1} / {epoch} error: {err}")



