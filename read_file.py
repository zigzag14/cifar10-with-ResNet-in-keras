
# coding: utf-8

# In[ ]:


import numpy as np
import pickle

# img_path = '../cifar-10-batches-py/'  #the file path of the training & testing data


# In[ ]:


def unpickle(file):      # load file with pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[ ]:


def convert_to_one_hot(Y, C):      #convert data to one hot form
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# In[ ]:


def img_read(img_path):
    X_train= np.zeros((0, 32, 32,3));
    Y_train= np.zeros((0, ),dtype="uint8");
    ## read the training batch one by one, concatenate them into X_train & Y_train

    num_batch= 5                         #number of training image batch, =5 in this case
    
    for index in range(1, num_batch+1):
        file_name= img_path+'data_batch_'+str(index)
        dict= unpickle(file_name)

        X= dict[b'data'];  # dim=[10000, 3072], The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue
        Y= dict[b'labels'];  # a list of length= 10000

        X= X.reshape((-1, 3, 32, 32)).transpose(0,2,3,1).astype("uint8")  #now the shape of X is (m, 32, 32, 3)
        Y= np.asarray(Y);
        X_train=np.concatenate((X_train, X), axis=0);
        Y_train=np.concatenate((Y_train, Y), axis=0);

    # now read the testing data    
    dict_test= unpickle(img_path+'test_batch')

    X_test= dict_test[b'data']
    Y_test= dict_test[b'labels']

    Y_test= np.asarray(Y_test)
    X_test= X_test.reshape((-1, 3, 32, 32)).transpose(0,2,3,1).astype("uint8")
    
    #preprocessing data as in the original resNet paper
    # 1. normalize X_train, X_test
    # 2. substract pixel mean
    # 3. convert to one hot

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    X_mean = np.mean(X_train, axis=0)     #remove pixel mean
    X_train -= X_mean
    X_test -= X_mean

    Y_train = convert_to_one_hot(Y_train, 10).T
    Y_test = convert_to_one_hot(Y_test, 10).T
    
    return (X_train, Y_train), (X_test, Y_test)

