import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
def test_examples(seq_size,n_points):

    seq = np.array(range(seq_size))
    t = 4*math.pi*seq/n_points
#    inputs = (np.sin(t) + 1)/2;
#    targets = (np.sin(2*t)+1)/2;
    inputs = np.sin(t);
    targets = np.sin(2*t);
    return inputs, targets, t

n_points = 60
seq_size = n_points+1
inputs,targets,t = test_examples(seq_size,n_points)
plt.plot(t,inputs)
#plt.show()
plt.plot(t,targets)
plt.show()
X_train = np.tile(inputs, (1000,1))
y_train = np.tile(targets,(1000,1))

data = {'X_train':X_train,'y_train':y_train}

path = '/home/visionteam/tf_tutorials/ECE_697_Fall2020/temporal_models/'
fname = 'data_fd.pkl'
file_handle = open(path+fname,'wb')
pickle.dump(data,file_handle)
file_handle.close()
print('saved  training data to:{}'.format(path+fname))

