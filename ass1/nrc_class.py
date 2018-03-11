import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from sklearn.model_selection import train_test_split

def split_data(data, labels):
    #shuffle data    
    xy_tuple = list(zip(data, labels))
    rnd.shuffle(xy_tuple)
    data = [x[0] for x in xy_tuple]
    labels = [x[1] for x in xy_tuple]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10) 
    n_train, n_features = np.shape(X_train)
    n_test = np.shape(X_test)[0]
    print(n_train, n_features, n_test)
    return X_train, X_test, y_train, y_test, n_train, n_features, n_test

def get_one_training(data, labels, index):
    X_train = [data[index], data[index + 554]]
    X_train = np.asarray(X_train)
    y_train = [labels[index], labels[index + 554]]
    y_train = np.asarray(y_train)
    X_test = []
    X_test.extend(data[index+1 : index+200])
    X_test.extend(data[index+1 + 554 : index+200+ 554])
    X_test = np.asarray(X_test)
    y_test = []
    y_test.extend(labels[index+1: index+200])
    y_test.extend(labels[index+1 + 554 : index+200+ 554])
    y_test = np.asarray(y_test)
    n_train, n_features = np.shape(X_train)
    n_test = np.shape(X_test)[0]
    # print(n_train, n_features, n_test)
    # print(y_test)
    return X_train, X_test, y_train, y_test, n_train, n_features, n_test
    
def ex2(data, labels):
    X_train, X_test, y_train, y_test, n_train, n_features, n_test = split_data(data, labels)
    NrC_class = NrC(X_train, y_train, n_train, n_features)
    NrC_class.gradient_descent()

    img0 = np.reshape(NrC_class.r[0], (8,8))
    img1 = np.reshape(NrC_class.r[1], (8,8))

    plt.figure(); plt.imshow(img0, cmap='gray'); 
    plt.figure(); plt.imshow(img1, cmap='gray'); 
    plt.show()#(block=False)

def ex3(data, labels):
    lambdas = [0, 0.1, 1, 10, 100, 1000]
    apparent_error = []
    true_error = []
    for lam in lambdas:
        train_err=0; test_err=0
        for index in range(100):
            X_train, X_test, y_train, y_test, n_train, n_features, n_test = get_one_training(data, labels, index)
            NrC_class = NrC(X_train, y_train, n_train, n_features, lam)
            tr, te = NrC_class.test_train_error(X_test, y_test)
            train_err+=tr; test_err+=te
        train_err/=100; test_err/=100
        apparent_error.append(train_err)
        true_error.append(test_err)
    plt.figure()
    plt.plot(lambdas, apparent_error,'-b', label = 'apparent error')
    plt.plot(lambdas, true_error,'-r', label = 'true error')
    plt.legend()
    plt.xlabel('lambda')
    # plt.xscale('logit')
    plt.show()

def check_classification(true_label, class_label):
    if true_label == class_label:
        return 0
    else: return 1

class NrC:
    def __init__(self, data, labels, n_train, n_features, 
        lambda_ = 1000, learning_rate = 0.01, seed = 0):
        self.data = data
        self.labels = labels
        self.n_features = n_features
        self.n_train = n_train
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        # self.rnd = np.random.mtrand.RandomState(seed)

        #initialize random initial weights (r in formula)
        self.r = []
        self.r.append(np.random.random((self.n_features))/1000)
        self.r.append(np.random.random((self.n_features))/1000)
        # print(len(self.r), len(self.r[0]), self.r)

        #calculate the amount of class elements
        self.n_class = []
        n_ones = np.sum(self.labels)
        self.n_class.append(len(self.labels) - n_ones)
        self.n_class.append(n_ones)
        # print(self.n_class[0], self.n_class[1])

    def gradient_descent(self):
        # print(self.lambda_)
        #stochastic gradient descent  
        if len(self.data) < 10:
            repeat = 1
        else: repeat = 1
        for r in range(repeat):
            # if r == 1: self.learning_rate = 0.0001
            for i in range(len(self.data)):
                self.input = self.data[i]
                # print(np.shape(self.input)) 
                grad0 = self.lambda_ * np.sign(self.r[0] - self.r[1])                           
                grad1 = -grad0
                
                if self.labels[i] == 0:             
                    grad0 += -2*(self.input - self.r[0])/ self.n_class[0]
                else:
                    grad1 += -2*(self.input - self.r[1])/ self.n_class[1]
                self.r[0]-= self.learning_rate * grad0                
                self.r[1]-= self.learning_rate * grad1
                # print(grad0)
                loss = self.loss_function(i)
        return loss
 
    def loss_function(self, index):
        if(self.labels[index] == 0):
            val = 0
        else:
            val = 1
        # print(self.input)
        # print(self.r[val])
        l_2 = np.sum(np.power(np.subtract(self.input, self.r[val]), 2)/self.n_class[val])
        l_1 = self.lambda_*np.sum(np.abs(np.subtract(self.r[0], self.r[1])))
        return l_1 + l_2

    # def loss(lambda_, input, r, r0, r1, n_class):
    #     l_2 = np.sum(np.power(np.subtract(input, r), 2)/n_class)
    #     l_1 = lambda_*np.sum(np.abs(np.subtract(r0, r1)))
    #     return l_1+l_2
    
    def test_train_error(self, X_test, y_test):
        self.gradient_descent()
        train_error = 0
        for i in range(len(self.data)):
            train_error += check_classification(self.labels[i], self.classify(self.data[i]))
        train_error/=len(self.data)

        test_error = 0
        for i in range(len(X_test)):
            # print(y_test[i])
            test_error += check_classification(y_test[i], self.classify(X_test[i]))
        test_error /= len(X_test)

        return train_error, test_error
    

    def classify(self, img):
        val0 = np.sum(np.power(np.subtract(img, self.r[0]), 2)/self.n_class[0])
        val1 = np.sum(np.power(np.subtract(img, self.r[1]), 2)/self.n_class[1])
        if val0 < val1: return 0
        else: return 1
        

if __name__ == '__main__':
    f = open('optdigitsubset.txt', 'r')
    lines = f.readlines()   
    digits=[]
    for line in lines:
        inner_list = [int(x) for x in line.split()]
        digits.append(inner_list)
    digits = np.asarray(digits)/255
    # img = digits[0].reshape(8,8)
    # plt.figure(); plt.imshow(img, cmap='gray'); 
    labels = np.empty(554, dtype=int)
    labels.fill(0)
    ones = np.empty(571, dtype=int)
    ones.fill(1)
    labels = np.append(labels, ones)

    ex2(digits, labels)
    # ex3(digits, labels)