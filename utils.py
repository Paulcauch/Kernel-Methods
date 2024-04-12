# UTILS

# IMPORT 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


# UTILS FUNCTIONS 


#######################################
#######################################
############## DATASET ################
#######################################
#######################################


def create_dataset_onevsall(Y,k):
    """  
    Transform the labels of class k in 1 and the labels of the others classes in -1 
    """

    Y_onevall=-np.ones_like(Y)
    Y_onevall[Y==k]=1

    return Y_onevall



def create_dataset_onevone(X,Y,k,j) :
    """
    Create a dataset X,Y, with two class k and j with label 1 and -1 
    """

    n_class1 = len(Y[Y==k])
    n_class2 = len(Y[Y==j])
    N=n_class1+n_class2

    y=-np.ones(N)
    y[:n_class1]=1

    x = np.zeros((N,X.shape[1]))
    x[:n_class1]=X[Y==k]
    x[n_class1:]=X[Y==j]

    return x,y



def create_test_set(X,Y,nbre):
    """
    Create a dataset X_train,Y_train,X_test,Y_test with nbre random test sample 
    """

    ind = np.random.choice(np.arange(len(X)),size=nbre,replace=False)

    y_test = Y[ind]
    x_test = X[ind]

    mask = np.ones(len(X), dtype=bool)
    mask[ind] = False 

    x_train = X[mask]
    y_train = Y[mask]

    return x_train,y_train,x_test,y_test




# Reduced Dataset 


def create_reduce_dataset(X,Y,size):
    """ 
    Create a dataset of size = 10*size with 10% of each class
    
    and transform the labels of class k in 1 and the labels of the others classes in -1 
    """

    Y_new=np.array([])
    X_new=np.random.randint(2,size=(1,X.shape[1]))

    for i in range(len(np.unique(Y))):
        ind=np.random.choice(np.array(np.where(Y==i))[0],size=size)
        X_new=np.concatenate((X_new,X[ind]))
        Y_new=np.concatenate((Y_new,i*np.ones(size)))
    X_new=X_new[1:]
    ind=np.arange(Y_new.shape[0])
    np.random.shuffle(ind)
    Y_shuffled=Y_new[ind]
    X_shuffled=X_new[ind]
    return X_shuffled,Y_shuffled


def create_reduce_dataset_onevsall(X,Y,k,size):
    """ 
    Create a dataset of size = 18*size with 50% of class k and 50% random classes  
    
    and transform the labels of class k in 1 and the labels of the others classes in -1 
    """

    Y_new=np.array([])
    X_new=np.random.randint(2,size=(1,X.shape[1]))

    for i in range(len(np.unique(Y))):
        if i == k :
            ind=np.random.choice(np.array(np.where(Y==i))[0],size=size*9)
        else :
            ind=np.random.choice(np.array(np.where(Y==i))[0],size=size)
        X_new=np.concatenate((X_new,X[ind]))
        if i==k:
            Y_new=np.concatenate((Y_new,np.ones(size*9)))
        else :
            Y_new=np.concatenate((Y_new,-np.ones(size)))
            
    X_new=X_new[1:]
    ind=np.arange(Y_new.shape[0])
    np.random.shuffle(ind)
    Y_shuffled=Y_new[ind]
    X_shuffled=X_new[ind]
    return X_shuffled,Y_shuffled





#######################################
#######################################
########### VIZUALISATION #############
#######################################
#######################################


def plot_images_grid(data, nrows, ncols):
    """
    Plot random image from the dataset
    """

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2, nrows*2))
    random=np.random.choice(data.shape[0],size=nrows*ncols)

    for j, ax in enumerate(axes.flat):

        i=random[j]

        if i < data.shape[0]:

            image_data = data[i, :]

            red_channel = image_data[:1024].reshape((32, 32))
            red_channel=(red_channel-red_channel.min())/(red_channel.max()-red_channel.min())
            green_channel = image_data[1024:2048].reshape((32, 32))
            green_channel=(green_channel-green_channel.min())/(green_channel.max()-green_channel.min())
            blue_channel = image_data[2048:].reshape((32, 32))
            blue_channel=(blue_channel-blue_channel.min())/(blue_channel.max()-blue_channel.min())

            image = np.stack((red_channel, green_channel, blue_channel), axis=-1)

            ax.imshow(image)
            ax.axis('off')

        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()



#######################################
#######################################
############# VALIDATION ##############
#######################################
#######################################


def accuracy(y_pred,y_test):
    """
    Compute the accuracy for two list of labels
    """
    acc = np.sum(y_pred==y_test)/len(y_pred)
    return acc


def commit(final_classes,name):
    """
    Create a csv file with the final results
    """
    df = pd.DataFrame({
    'Id': range(1, 2001),  
    'Prediction': final_classes  })

    df.to_csv(f'{name}.csv', index=False)