import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import theano
from theano.ifelse import ifelse
import theano.tensor as T

def load_data(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, 
                                                        labels, 
                                                        test_size=0.25, 
                                                        random_state=42)
    
    n_train, n_features = np.shape(x_train)
    n_test = np.shape(x_test)[0]
    
    def shared_data(x, y):
        shared_data = theano.shared(np.asarray(x, dtype=theano.config.floatX))
        shared_label = theano.shared(np.asarray(y, dtype=theano.config.floatX))
        
        return shared_data, T.cast(shared_label, 'int32')
    
    train_data, train_label = shared_data(x_train, y_train)
    test_data, test_label = shared_data(x_test, y_test)
    
    return (train_data, test_data), (train_label, test_label), (n_train, n_test), n_features

class NRC(object):
    
    def __init__(self, data, labels, n_features, l1, learning_rate=0.01, seed=0):
        
        self.data = data
        self.labels = labels
        self.l1 = l1
        self.seed = seed
        self.learning_rate = learning_rate
        self.rng = np.random.mtrand.RandomState(self.seed)
        
        self.w = []
        
        # TODO: generalise to multiple classes
        for i in range(2):
            self.init_weights(n_features, i)
        
        self.n_class = []
        # TODO: generalise to multiple classes
        for i in range(2):
#            self.n_class.append(np.shape(self.labels[0]))
            self.n_class.append(T.shape(T.nonzero(T.eq(self.labels, i))[0]))
    
    def init_weights(self, n_features, label):
        
        rand_range = 0.01
        w = theano.shared(value=np.asarray(self.rng.uniform(low=-rand_range, 
                                                            high=rand_range, 
                                                            size=(1, n_features)), 
                                           dtype=theano.config.floatX), 
                                           name='w{}'.format(label))
        self.w.append(w)
    
    def update_learner(self):
        
        index = T.iscalar(name='index')
        self.input = self.data[index:index+1]
        
        prev_cost = theano.shared(value=np.asarray(0))
        
        cost_l2, cost_l1 = self.loss(index)
        cost = cost_l2 + cost_l1 + prev_cost
        prev_cost = cost_l2
        
        updates = []
        for i in range(2):
            grad = T.grad(cost, self.w[i])
            new_weight = self.w[i] - self.learning_rate*grad
            updates.append((self.w[i], new_weight))
        
        self.learn = theano.function(name='learn',
                                     inputs=[index],
                                     outputs=[cost],
                                     updates=updates)
    
    def loss(self, index):
        
        l2_term = T.switch(T.eq(self.labels[index], 0),
                           T.sum(T.pow(self.input - self.w[0], 2) / self.n_class[0][0]),
                           T.sum(T.pow(self.input - self.w[1], 2) / self.n_class[1][0]))
        
#        if T.eq(self.labels[index], 1):
#            l2_term = T.sum(T.pow(self.input - self.w[1], 2) / self.n_class[1][0])
#        elif T.eq(self.labels[index], 0):
#            l2_term = T.sum(T.pow(self.input - self.w[0], 2) / self.n_class[0][0])
        
        l1_term = self.l1*T.sum(T.abs_(self.w[0] - self.w[1]))
        
        
        return l2_term, l1_term

if __name__ == '__main__':
    data = np.load('data/digits.npy')
    # scale values
    data = data/255
    lab = np.load('data/labels.npy')
    lab = lab.astype(int)
    
    data_shared, label_shared, n_shared, n_features = load_data(data, lab)
    
    train_data, test_data = data_shared
    train_label, test_label = label_shared
    n_train, n_test = n_shared
    
    l1 = 1000
    learning_rate = 0.1
    
    nrc_learner = NRC(train_data, train_label, n_features, l1, learning_rate=learning_rate)
    nrc_learner.update_learner()
    
    for i in range(n_train):
        cost = nrc_learner.learn(i)
        print('Index {}, cost {}'.format(i, cost[0]))
    
    w0 = np.asarray(nrc_learner.w[0].eval())
    w1 = np.asarray(nrc_learner.w[1].eval())
    
    img0 = np.reshape(w0, (8, 8))
    img1 = np.reshape(w1, (8, 8))
    
    plt.figure(); plt.imshow(img0, cmap='gray'); plt.show(block=False)
    plt.figure(); plt.imshow(img1, cmap='gray'); plt.show(block=False)
