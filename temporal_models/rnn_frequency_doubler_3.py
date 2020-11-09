"This program was adapted by Ruthvik Vaila from code written by Denny Britz "
" See http://www.wildml.com "
#http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
#
import numpy as np
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import time
import sys

path = '/home/visionteam/tf_tutorials/ECE_697_Fall2020/data/'
fname = 'data_fd2.pkl'
file_handle = open(path+fname,'rb')
data = pickle.load(file_handle, encoding='latin1')
file_handle.close()

print('data.keys() =  {}'.format(data.keys()))


class RNNNumpy:
    def __init__(self, output_dim, input_dim, hidden_dim=100, bptt_truncate=0):
        # Assign instance variables
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.Wx = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
        self.Wy = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))
        self.Wh = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    def forward_propagation(self, x):   # x is X_train[i]
        T = len(x)  # The total number of time steps
        # Save all hidden states h[t] for back prop.
        # One additional row for h[0] = 0
        h = np.zeros((T + 1, self.hidden_dim))  # h.shape = (T+1,nh)
        # Save all outputs for back prop.
        y_hat = np.zeros((T, self.output_dim))   # y_hat.shape = (T,no)   
        for t in np.arange(T):
#           # Wx.dot(x[t]).shape = (nh,1)
            # x[t].shape = () is a scalar, Wh.dot(h[t-1]).shape is (nh,)
            h[t] = np.tanh(self.Wx.dot(x[t]).flatten() + self.Wh.dot(h[t-1]))
            y_hat[t] = self.Wy.dot(h[t])
        return [y_hat, h]
    def predict(self, x):  # x is row vector of dimension of T
        # Perform forward propagation and return index of the highest score
        y_hat, h = self.forward_propagation(x) 
        return y_hat 

    def calculate_total_loss(self, x, y):  # Here x is X_train, y = y_train
        C = 0
        # For each sentence...
        for i in np.arange(len(y)):
            
            y_hat, h = self.forward_propagation(x[i])
            #print ((y_hat.flatten()-y[i])**2).sum()
            #print(y_hat.shape,len(y[i]))
            error = ((y_hat.flatten()-y[i])**2)/2.0
            #print error.sum(),i
            # Add to the loss based on how off we were
            C +=  np.sum(error)
        #print C
        return C
 
    def calculate_loss(self, x, y):  #  x is X_train, y is y_train
        # Divide the total loss by the number of training examples
        #N = np.sum((len(y_i) for y_i in y))
        N = y.size
        return self.calculate_total_loss(x,y)/N

    def update_loss(self, x, y):  # Here x is X_train[i], y = y_train[i]
        Cbptt = 0
        y_hat, h = self.forward_propagation(x)
        error = (1/2)*(y_hat-y)**2
        Cbptt =  np.sum(error)
        return Cbptt

    def bptt(self, x, y):  # Here x = X_train[i], y = y_train[i] 
        T = len(y)
        # Perform forward propagation  y_hat.shape = (T,nI), h.shape=(T+1,nh)
        y_hat, h = self.forward_propagation(x) 
        # We accumulate the gradients in these variables
        dCdWx = np.zeros(self.Wx.shape)
        dCdWy = np.zeros(self.Wy.shape)
        dCdWh = np.zeros(self.Wh.shape)
        # y.shape = (T,) while y_hat.shape = (T,nI)
        y_hat = y_hat.flatten() # y_hat now has shape (T,)
        dCdy = -1*(y-y_hat)   #  dCdy.shape = (T,) 
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dCdWy += np.outer(dCdy[t], h[t].T)
            # Initial dCdz_t calculation
            dCdz_t = self.Wy.T.dot(dCdy[t]) * (1 - (h[t] ** 2).reshape(self.hidden_dim,1))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                #print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dCdWh += np.outer(dCdz_t, h[bptt_step-1])
                #dCdWx[:,x[bptt_step]] += dCdz_t
                #dCdWx += dCdz_t
                dCdWx += np.outer(dCdz_t, x[bptt_step])
                # Update dCdz_t for next step
                dCdz_t = self.Wh.T.dot(dCdz_t) * (1 - h[bptt_step-1] ** 2).reshape(self.hidden_dim,1)
        return [dCdWx, dCdWy, dCdWh]

    def numpy_sdg_step(self, x, y, learning_rate): # Here x is X_train[i] and y = y_train[i]
        # Calculate the gradients
        dCdWx, dCdWy, dCdWh = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.Wx -= learning_rate * dCdWx
        self.Wy -= learning_rate * dCdWy
        self.Wh -= learning_rate * dCdWh

# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
    # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
#            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            print("Loss after num_examples_seen=%d epoch=%d: %f" % (num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
                      
#np.random.seed(10)
model = RNNNumpy(output_dim=1, input_dim=1, hidden_dim=10, bptt_truncate=10)
X_train = data['X_train']
y_train = data['y_train']
print('Shape of the data:{}'.format(X_train.shape))
losses = train_with_sgd(model, X_train, y_train, nepoch=5, evaluate_loss_after=1)
y_hat, h = model.forward_propagation(X_train[10])
#print(y_hat)
#print(X_train[10])
Wh = model.Wh
#print('Wh=\n {}'.format(Wh))

plt.plot(range(X_train.shape[-1]),X_train[10],label='$x(t)$')
plt.plot(range(X_train.shape[-1]),y_hat,label='$\hat{y}(t)$')
plt.plot(range(X_train.shape[-1]),y_train[10],label='$y(t)$')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(prop={'size': 14})
plt.suptitle('$\\ RNN \\ to \\ transform \\ \\sin(t) \\ to \\ sin(2t) $',fontsize=22)
plt.savefig('sinusoid1.png')
plt.show()

error = y_hat - y_train[10].reshape(X_train.shape[-1],1)
#plt.plot(range(31),X_train[10],label='$x(t)$')
plt.plot(range(X_train.shape[-1]),error,label='$y(t) - \hat{y}(t)$')
#plt.plot(range(31),y_train[10],label='$y(t)$')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(prop={'size': 14})
plt.suptitle('error(t) = $y(t) - \hat{y}(t)$',fontsize=22)
plt.savefig('sinusoid1.png')
plt.show()





#plt.plot(range(31),X_train[10],label='$x(t)$')
#plt.plot(range(31),y_hat,label='$y(t)$')
#plt.plot(range(31),y_train[10],label='$d(t)$')
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#plt.legend(prop={'size': 14})
#plt.suptitle('$\\ RNN \\ to \\ transform \\ \\sin(t) \\ to \\ sin(2t) $',fontsize=22)
#plt.savefig('sinusoid2.png')



