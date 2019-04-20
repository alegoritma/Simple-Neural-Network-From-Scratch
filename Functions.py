import numpy as np

class functions:
    #sigmoid
    def sigmoid(s):
        return 1/(1+np.exp(-s))

    def sigmoid_derv(s):
        return (1-s)*s

    #softmax
    def softmax(s):
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    #tanh
    def tanh(s):
        return np.tanh(s)

    def tanh_derv(s):
        return (1 - (s ** 2))

    #squash
    def squash(s):
        for i in range(0, len(s)):
            for k in range(0, len(s[i])):
                s[i][k] = (s[i][k]) / (1 + abs(s[i][k]))
        return s

    def squash_derv(s):
        for i in range(0, len(s)):
            for k in range(0, len(s[i])):
                if s[i][k] > 0:
                    s[i][k] = (s[i][k]) / (1 + s[i][k])
                else:
                    s[i][k] = (s[i][k]) / (1 - s[i][k])
        return s

    #gaussian
    def gaussian(s):
        for i in range(0, len(s)):
            for k in range(0, len(s[i])):
                s[i][k] = np.exp(-s[i][k] ** 2)
        return s

    def gaussian_derv(s):
        for i in range(0, len(s)):
            for k in range(0, len(s[i])):
                s[i][k] = -2* s[i][k] * np.exp(-s[i][k] ** 2)
        return functions.gaussian(s)

    def relu(s):
        for i in range(0, len(s)):
            for k in range(0, len(s[i])):
                if s[i][k] > 0:
                    pass  # do nothing since it would be effectively replacing x with x
                else:
                    s[i][k] = 0
        return s
    def relu_derv(s):
        for i in range(0, len(s)):
            for k in range(len(s[i])):
                if s[i][k] > 0:
                    s[i][k] = 1
                else:
                    s[i][k] = 0
        return s
    #cost functions
    def delta_cross_entropy(pred,real):
        n_samples = real.shape[0]
        res = pred-real
        return res/n_samples

    #loss functions
    def cross_entropy_loss(pred,real):
        n_samples = real.shape[0]
        logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
        loss = np.sum(logp)/n_samples
        return loss

act_functs = {"sigmoid": functions.sigmoid, "softmax": functions.softmax, "tanh": functions.tanh, "squash": functions.squash, "gaussian": functions.gaussian, "relu": functions.relu}
act_derv_functs = { "sigmoid": functions.sigmoid_derv, "softmax": functions.sigmoid_derv, "tanh": functions.tanh_derv, "squash": functions.squash_derv, "gaussian":functions.gaussian_derv, "relu": functions.relu_derv}
cost_functs = {"cross_entropy": functions.delta_cross_entropy}
loss_functs = {"cross_entropy": functions.cross_entropy_loss}
