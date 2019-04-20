import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import struct as st
from tqdm import tqdm
from Functions import *

class Dense:
    def __init__(self,ip_dim, neurons, activation):
        self.w = np.random.randn(ip_dim, neurons)
        self.b = np.zeros((1,neurons))
        self.activation = act_functs[activation]
        self.act_derv = act_derv_functs[activation]
        self.z = None
        self.a = None
        self.a_delta = None
        self.z_delta = None


class Sequential:
    def __init__(self, lr=0.5, ip_dim=None, op_dim=None):
        self.lr = lr
        self.layers = []
        self.ip_dim = ip_dim
        self.op_dim = op_dim
        self.compiled = False
        self.a_delta_output = []
    def add_hidden_dense(self, n_cells, activation="sigmoid"):
        self.layers.append(Dense(ip_dim=self.ip_dim,
                                neurons=n_cells,
                                activation=activation))
        self.ip_dim = n_cells

    def compile(self, loss="cross_entropy", cost="cross_entropy"):
        self.layers.append(Dense(ip_dim=self.ip_dim,
                                    neurons=self.op_dim,
                                    activation="softmax"))

        self.loss_funct = loss_functs[loss]
        self.cost_function = cost_functs[cost]
        self.compiled = True

    def feedforward(self,x,y=None):
        # a holds scores in current layer's perceptrons
        self.x = x
        self.a = x
        self.y = y
        for i in range(len(self.layers)):
            self.layers[i].z = np.dot(self.a, self.layers[i].w) + self.layers[i].b
            self.layers[i].a = self.layers[i].activation(self.layers[i].z)
            self.a = self.layers[i].a
        self.acc_list.append(self.accuracy(self.a,self.y))

    def backprop(self):
        loss = self.loss_funct(self.a,self.y)
        # print('Error:', loss)
        self.loss_list.append(loss) # save for graph

        #output layer
        self.layers[len(self.layers)-1].a_delta = self.cost_function(self.a,self.y)
        self.a_delta_output.append(pd.DataFrame(self.layers[len(self.layers)-1].a_delta).T[0].sum())
        for i in reversed(range(len(self.layers)-1)):
            self.layers[i].z_delta = np.dot(self.layers[i+1].a_delta, self.layers[i+1].w.T)
            self.layers[i].a_delta = self.layers[i].z_delta * self.layers[i].act_derv(self.layers[i].a)

        #gredient descent
        #   -output layer
        self.layers[len(self.layers)-1].w -= self.lr * np.dot(self.layers[i-2].a.T, self.layers[len(self.layers)-1].a_delta)
        self.layers[len(self.layers)-1].b -= self.lr * np.sum(self.layers[len(self.layers)-1].a_delta, axis=0, keepdims=True)

        #   -hidden self.layers
        for i in reversed(range(len(self.layers)-2)):
            self.layers[i+1].w -= self.lr * np.dot(self.layers[i].a.T, self.layers[i+1].a_delta)
            self.layers[i+1].b -= self.lr * np.sum(self.layers[i+1].a_delta, axis=0)

        #   -first hidden layer
        self.layers[0].w -= self.lr * np.dot(self.x.T, self.layers[0].a_delta)
        self.layers[0].b -= self.lr * np.sum(self.layers[0].a_delta, axis=0)

    def predict(self,x):
        self.a = x
        for i in range(len(self.layers)):
            self.layers[i].z = np.dot(self.a, self.layers[i].w) + self.layers[i].b
            self.layers[i].a = self.layers[i].activation(self.layers[i].z)
            self.a = self.layers[i].a
        return self.a

    def fit(self,x=None, y=None, batch_size=32, epochs=1):
        assert (x.shape[0] == y.shape[0]), "Input size doesn't match with output's"
        self.loss_list = []
        self.acc_list = []
        n_samples = x.shape[0]
        last_batch = n_samples%batch_size
        iters = int(n_samples/batch_size)
        for _ in range(epochs):
            for i in tqdm(range(iters)):
                self.feedforward(x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
                self.backprop()
            if last_batch:
                self.feedforward(x[-last_batch:],y[-last_batch:])
                self.backprop()
            print(f"acc: {self.acc_list[-1]} || loss: {self.loss_list[-1]}")

    def accuracy(self, pred=None, real=None):
        trues = len([true for true in (pred.argmax(axis=1) == real.argmax(axis=1)) if true])
        all = len(pred.argmax(axis=1) == real.argmax(axis=1))
        return trues/all

    def graph_acc(self):
        plt.plot([i for i in range(len(self.acc_list))],self.acc_list)
        plt.show()
    def graph_loss(self):
        plt.plot([i for i in range(len(self.loss_list))],self.loss_list)
        plt.show()




def launch(train_x,train_y,test_x,test_y,n_ns=[64,64],batch_size=64, epochs=1):
    model = Sequential(lr=0.1,ip_dim=784, op_dim=10)
    for n_n in n_ns:
        model.add_hidden_dense(n_cells=n_n, activation="sigmoid")
    model.compile()
    model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs)
    predicted = model.predict(test_x)
    overallacc = model.accuracy(pred=predicted, real=test_y)
    return model, overallacc




















#
