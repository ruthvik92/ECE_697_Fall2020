import sys
print(sys.version)
import numpy as np
import h5py, sys, os, time, pickle, gzip
GPU =  True
if(not GPU):
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import matplotlib.pyplot as plt
from keras import backend
import IPython
from tensorflow.python.client import device_lib
import nn_classifierclass as cls
print(device_lib.list_local_devices())


filename = 'data/emnist-balanced.pkl.gz'
filehandle = gzip.open(filename, 'rb')
train_data, val_data, test_data = pickle.load(filehandle)
filehandle.close()
train_x, train_y = train_data
print('Train data shape:{} and labels shape:{}'.format(train_x.shape, train_y.shape))
val_x, val_y = val_data
print('Valid data shape:{} and labels shape:{}'.format(val_x.shape, val_y.shape))
## combine train and validation data, classifier_class can split it inside 
train_x = np.concatenate([train_x, val_x], axis=0)
train_y = np.concatenate([train_y, val_y], axis=0)
print('Train data shape:{}'.format(train_x.shape))
print('Train labels shape:{}'.format(train_y.shape))
test_x, test_y = test_data
print('Test data shape:{}'.format(test_x.shape))
print('Test labels shape:{}'.format(test_y.shape))


n_classes = 47
n_hidden = 1
network_structure = [train_x.shape[1],1500,n_classes] ##will be ignored, only sigmoid neurons
#activation_fns = ['sigmoid']*(n_hidden)+['softmax']
#activation_fns = ['tanh']*(n_hidden)+['softmax']
#activation_fns = ['relu']*(n_hidden)+['softmax']
activation_fns = ['swish']*(n_hidden)+['softmax'] #will be ignored, only sigmoid neurons and last layer is softmax
#activation_fns = ['softmax']
#sys.exit()
#weight_init = 'he_uniform'
weight_init = 'glorot_uniform'
eta_drop_type = 'plateau'
lmbda = 0.000
batch_size = 32
eta = 0.005

log_path = '/home/visionteam/tf_tutorials/logs/'+''.\
join(activation_fns+['-',weight_init,'-',eta_drop_type,str(-lmbda)])+'/eta'+str(-eta)
print(log_path)
repeats = 1
all_histories = []
for repeat in range(repeats):
    print('Repeat:{}'.format(repeat))
    backend.clear_session()
    neural_net = cls.Classifier(train_data=(train_x,train_y),
                                test_data=(test_x,test_y),
                                network_structure=network_structure,activation_fns=activation_fns,
                                epochs=1,eta=eta,lmbda=lmbda,verbose=1,plots=False,optimizer='adam',
                                eta_decay_factor=1.007,patience=8,eta_drop_type=eta_drop_type,
                                epochs_drop=1, val_frac=0.09,drop_out=0.0,ip_lyr_drop_out=0.0,
                                leaky_alpha=0.1,leaky_relu=False,weight_init=weight_init,
                                bias_init=0.1,batch_size=batch_size,log_path=log_path)
    neural_net.numpy_fcn_classifier()
    all_histories.append(neural_net.history)
