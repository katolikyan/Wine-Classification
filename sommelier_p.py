class perceptron:

    import random

    def __init__(self, num):
        self.w = list(map(lambda x: self.random.uniform(-1, 1), range(num)))
        self.b = self.random.uniform(-1, 1)
        self.log = []

    def activation(self, s):
        if s > 0:
            return 1
        else:
            return -1

    def summatory(self, x, b):
        summ = 0
        for i in range(len(self.w)):
            summ += self.w[i] * x[i]
        summ += self.b
        return summ

    def learning_simple(self, data, y, ep_num, learning_rate):
        y_out = []
        epoch = 1

        # training loop:
        while True:
            errors = 0
            learned = 1
            y_out[:] = []   # clean output;
            # get output:
            for x in data:
                y_out.append(self.activation(self.summatory(x, self.b))) # get output for each vine;
            #learn:
            for i in range(len(y)):
                if y[i] != y_out[i]: # compare output with expected;
                    errors += 1
                    learned = 0
                    for j in range(len(self.w)):
                        self.w[j] += learning_rate * (y[i] - y_out[i]) * data[i][j] # balance weights for each parametr;
                    self.b += learning_rate * (y[i] - y_out[i])                                     # balance bias
            self.log.append((epoch, errors, self.w.copy(), self.b))
            if (epoch == ep_num) or learned:
                break
            epoch += 1

class adaline:

    import random
    import math

    def __init__(self, num):
        self.w = list(map(lambda x: self.random.uniform(-1, 1), range(num)))
        self.b = self.random.uniform(-1, 1)
        self.log = []

    def activation(self, s):
        return (1 if s > 0 else -1)
        #return 1 / (1 + self.math.exp(-(s)))

    def summatory(self, x):
        summ = 0
        for i in range(len(self.w)):
            summ += self.w[i] * x[i]
        summ += self.b
        return summ

    def target(self, y_out, y):
        return sum([(y[i] - y_out[i]) ** 2 for i in range(len(y))]) / 2.0

    def gdl(self, data, y, ep_num, learning_rate, theta):
        y_out = []
        epoch = 1;

        # training loop:
        while True:
            learned = 1
            y_out[:] = []   # clean output;

            # get output:
            for x in data:
                y_out.append(self.summatory(x)) # get output for each vine;
            #learn:
            if self.target(y_out, y) > theta:
                learned = 0
                for i in range(len(y)):
                    for j in range(len(self.w)):
                        self.w[j] += learning_rate * (y[i] - y_out[i]) * data[i][j] # balance weights for each parametr;
                    self.b += learning_rate * (y[i] - y_out[i])
            self.log.append((epoch, self.target(y_out, y), self.w.copy(), self.b))
            if (epoch == ep_num) or learned:
                break
            epoch += 1


    def gdl_online(self, data, y, ep_num, learning_rate, theta):
        y_out = []
        epoch = 1;

        # training loop:
        while True:
            learned = 1
            y_out[:] = []   # clean output;

            # get output:
            for x in data:
                y_out.append(self.summatory(x)) # get output for each vine;
            #learn:
            if self.target(y_out, y) > theta:
                learned = 0
                for i in range(len(y)):
                    for j in range(len(self.w)):
                        self.w[j] += learning_rate * (y[i] - self.summatory(data[i])) * data[i][j] # balance weights for each parametr;
                    self.b += learning_rate * (y[i] - y_out[i])

            self.log.append((epoch, self.target(y_out, y), self.w.copy(), self.b))
            if (epoch == ep_num) or learned:
                break
            epoch += 1

    def predict(self, data):
        predictions = []
        for x in data:
            predictions.append(self.activation(self.summatory(x)))
        return (predictions)
