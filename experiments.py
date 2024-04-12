# EXPERIMENT FUNCTION

#IMPORT
from Algortihms import *
from utils import *
from Kernels import * 
from tqdm import tqdm




#######################################
#######################################
############# ONE VS ONE  #############
#######################################
#######################################


# Hyperparameter tuning with SVM (One v one)

def onevone_test_param_SVM(X,Y,list_C,list_param,ker):
    """
    Return the accuracy of SVM onevone method for a list of parameters
    """
    y_preds=[]
    accuracys=[]
    Models={}
    K=10
    nbre_modele=int(K*(K-1)/2)
    
    x_train,y_train,x_test,y_test = create_test_set(X,Y,nbre=500)

    for C in list_C : 
        for param in list_param :

            #TRAIN

            for i in range(K):

                for j in range(i+1,K):
                    
                    print(i,j)
                    x,y = create_dataset_onevone(x_train,y_train,i,j)

                    if ker == 'RBF':
                        kernel = RBF(sigma=param).kernel
                    elif ker == 'poly':
                        kernel = polynomial(d=param).kernel

                    mod = KernelSVC(C = C,kernel = kernel)
                    
                    mod.fit(x,y)

                    Models[(i, j)] = mod


            #TEST

            votes = np.zeros((len(x_test), K))

            for (class1,class2), model in Models.items():

                predictions = model.predict(x_test)

                for i, prediction in enumerate(predictions):
                    if prediction == 1:
                        votes[i, class1] += 1  # La première classe du tuple reçoit un vote
                    else:
                        votes[i, class2] += 1  # La seconde classe du tuple reçoit un vote

            final_classes = np.argmax(votes,axis=1)
            y_preds.append(final_classes)

            acc = accuracy(final_classes,y_test)
            if ker == 'RBF':
                print(f'Accuracy : {acc} | C = {C} | sigma = {param}')
            elif ker == 'poly':
                print(f'Accuracy : {acc} | C = {C} | d = {param}')
            accuracys.append(acc)
    
    return accuracys,y_preds
        
        


# Hyperparameter tuning with Polynomial Kernel and Logistic Regression (One v one)


def onevone_test_param_log(X,Y,list_C,list_param,ker):
    """
    Return the accuracy of the Logistic reg onevone method for a list of parameters
    """
    y_preds=[]
    accuracys=[]
    Models={}
    K=10
    nbre_modele=int(K*(K-1)/2)
    
    x_train,y_train,x_test,y_test = create_test_set(X,Y,nbre=500)

    for C in list_C : 
        for param in list_param :

            #TRAIN

            for i in range(K):

                for j in range(i+1,K):
                    
                    print(i,j)
                    x,y = create_dataset_onevone(x_train,y_train,i,j)

                    if ker == 'RBF':
                        kernel = RBF(sigma=param).kernel
                    elif ker == 'poly':
                        kernel = polynomial(d=param).kernel

                    mod = KernelLogisticRegression(kernel,reg_param=C)
                    
                    mod.fit(x,y)

                    Models[(i, j)] = mod


            #TEST

            votes = np.zeros((len(x_test), K))

            for (class1,class2), model in Models.items():

                predictions = model.predict(x_test)
                predictions[predictions>0.5]=1

                for i, prediction in enumerate(predictions):
                    if prediction == 1:
                        votes[i, class1] += 1  # La première classe du tuple reçoit un vote
                    else:
                        votes[i, class2] += 1  # La seconde classe du tuple reçoit un vote

            final_classes = np.argmax(votes,axis=1)
            y_preds.append(final_classes)

            acc = accuracy(final_classes,y_test)
            if ker == 'RBF':
                print(f'Accuracy : {acc} | C = {C} | sigma = {param}')
            elif ker == 'poly':
                print(f'Accuracy : {acc} | C = {C} | d = {param}')
            accuracys.append(acc)
    
    return accuracys,y_preds
        





# PCA + SVM 



def onevone_test_param_PCA_SVM(X,Y,list_C,list_param,ker,dim):
    """
    Return the accuracy of the SVM combined with the PCA onevone method for a list of parameters
    """
    y_preds=[]
    accuracys=[]
    Models={}
    K=10
    nbre_modele=int(K*(K-1)/2)


    kernel=RBF().kernel
    PCA=KernelPCA(kernel=kernel,r=500)
    PCA.compute_PCA(X) # We choose the most dim of X
    Xtrans=PCA.transform(X)

    x_train,y_train,x_test,y_test = create_test_set(Xtrans,Y,nbre=dim)


    for C in list_C : 
        for param in list_param :

            #TRAIN

            for i in range(K):

                for j in range(i+1,K):
                    
                    print(i,j)
                    x,y = create_dataset_onevone(x_train,y_train,i,j)

                    if ker == 'RBF':
                        kernel = RBF(sigma=param).kernel
                    elif ker == 'poly':
                        kernel = polynomial(d=param).kernel

                    mod = KernelSVC(C = C,kernel = kernel)
                    
                    mod.fit(x,y)

                    Models[(i, j)] = mod


            #TEST

            votes = np.zeros((len(x_test), K))

            for (class1,class2), model in Models.items():
                
                predictions = model.predict(x_test)

                for i, prediction in enumerate(predictions):
                    if prediction == 1:
                        votes[i, class1] += 1  # La première classe du tuple reçoit un vote
                    else:
                        votes[i, class2] += 1  # La seconde classe du tuple reçoit un vote

            final_classes = np.argmax(votes,axis=1)
            y_preds.append(final_classes)

            acc = accuracy(final_classes,y_test)
            if ker == 'RBF':
                print(f'Accuracy : {acc} | C = {C} | sigma = {param}')
            elif ker == 'poly':
                print(f'Accuracy : {acc} | C = {C} | d = {param}')
            accuracys.append(acc)
    
    return accuracys,y_preds
        








#######################################
#######################################
############# ONE VS ALL  #############
#######################################
#######################################


def onevall_test_param_svm(X,Y,list_C,list_param,ker):
    """
    Return the accuracy of the SVM onevall method for a list of parameters
    """
    y_preds=[]
    accuracys=[]
    Models = []

    x_train,y_train,x_test,y_test = create_test_set(X,Y,nbre=2000)

    for C in list_C : 
        for param in list_param :
    #TRAIN

            for j in tqdm(range(10)):

                y = create_dataset_onevsall(y_train,k=j)
                
                if ker == 'RBF':
                    kernel = RBF(sigma=param).kernel
                elif ker == 'poly':
                    kernel = polynomial(d=param).kernel
                classifier=KernelSVC(C=0.08,kernel=kernel)
                
                classifier.fit(x_train,y)
                
                # Stocker le modèle SVM entraîné
                Models.append(classifier)

            scores = np.zeros((10,x_test.shape[0]))

            # TEST

            for j, log_model in enumerate(Models):
                # Prédire le score pour la classe actuelle
                score = log_model.separating_function(x_test)
                
                # Stocker le score pour la classe actuelle
                scores[j] = score

            final_classes = np.argmax(scores,axis=0)
            y_preds.append(final_classes)

            acc=accuracy(final_classes,y_test)
            if ker == 'RBF':
                print(f'Accuracy : {acc} | C = {C} | sigma = {param}')
            elif ker == 'poly':
                print(f'Accuracy : {acc} | C = {C} | d = {param}')
            accuracys.append(acc)

    return accuracys,y_preds
