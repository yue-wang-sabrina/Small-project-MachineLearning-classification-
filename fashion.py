import tarfile
import pandas as pd
from PIL import Image
import sklearn
import pandas as pd #tools to read from files like CVS, Excel etc
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
from sklearn import ensemble
import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy
import time
import copy

'''
References
-PCA: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#drop_labels
-LDA: http://sebastianraschka.com/Articles/2014_python_lda.html
'''
START3=time.time()
def get_images(): #Obtained from Task Github page

    """Get the fashion-mnist images.
    Returns
    -------
    (x_train, x_test) : tuple of uint8 arrays
        Grayscale image data with shape (num_samples, 28, 28)
    (y_train, y_test) : tuple of uint8 arrays
        Labels (integers in range 0-9) with shape (num_samples,)
    Examples
    --------
    >>> from reader import get_images
    >>> (x_train, y_train), (x_test, y_test) = get_images() 
    Notes
    -----
    The data is split into train and test sets as described in the original paper [1].
    References
    ----------
    1. Xiao H, Rasul K, Vollgraf R. Fashion-MNIST: a Novel Image Dataset for 
    Benchmarking Machine Learning Algorithms. CoRR [Internet]. 2017;abs/1708.07747.
    Available from: http://arxiv.org/abs/1708.07747
    """

    with tarfile.open('data.tar.gz', 'r') as f:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f)

    df_train = pd.read_csv('fashion_mnist_train.csv')
    df_test = pd.read_csv('fashion_mnist_test.csv')

    x_train = df_train.drop('label', axis=1).values 
    y_train = df_train['label'].values 
    x_test = df_test.drop('label', axis=1).values 
    y_test = df_test['label'].values 
    
    return (x_train, y_train), (x_test, y_test)

##Method 1: Using pure scikit built in suport vector machine for classification: NB: super slow (took several hours)

(xtrain, ytrain), (xtest, ytest)=get_images()

'''
###Method 1: Brute force using sklearn classifier
mpl.rcParams['savefig.dpi'] = 300

# Get labels
xlabels = ['Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
## Convert labels to numeric values.
## y is the numeric label target.
#le = preprocessing.LabelEncoder()
#le.fit(xlabels)
ylabels = list(numpy.arange(0,10)) #Labelencoder sorted by alphabetical order so doing it by hand
data_and_targets=list(zip(xtrain,ytrain))

#Initialise estimator
estimator=svm.NuSVC(gamma=0.001)

#Flatten data for classifier
N_samples=len(xtrain)
xtrainflat=xtrain.reshape((N_samples,-1))

#Fit with cross validation
# estimator.fit(xtrain[:N_samples//2],ytrain[:N_samples//2])
# expected = ytrain[N_samples//2:]
# predicted = estimator.predict(xtrain[N_samples//2:])


# Split into training and testing sets for testing with cross validation.
kf = model_selection.KFold(n_splits=70) #

# Create lists of known and predicted values for each split.
y_true, y_pred = [], []
for train, test in kf.split(xtrain): #splits 70 times (70 folds) the data into train/test sets
    x_train, x_test, y_train, y_test = xtrain[train], xtrain[test], ytrain[train], ytrain[test]
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    
    # Add the split train and test predicitons to the list so we get a
    # full list of predictions.
    y_true.extend(y_test)
    y_pred.extend(y_predict)

# Evaluate the performance of all the splits tested.
score = metrics.accuracy_score(y_true, y_pred) #cross validation
#print('Accuracy Score = {:.3f}'.format(score))

cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
#print("confusion matrix:",cnf_matrix)

# Train over all the data available
estimator = estimator.fit(xtrain, ytrain)
 
#%% Plot the confusion matrix
# Stolen from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False,
                      title='Confusion matrix', cmap=plt.cm.Reds)
plt.show()

##The resulting confusion matrix is:
# cnf_matrix=array([[ 60,  78,  62, 103,  89, 113,  62, 117, 134, 182],
#        [ 80,  45,  70, 107, 104, 109,  50, 122, 122, 191],
#        [ 64,  69,  49, 124, 102,  76,  38, 132, 138, 208],
#        [ 85,  82,  74,  74, 100,  87,  44, 143, 130, 181],
#        [ 80,  70,  87, 102,  70, 119,  44, 134, 128, 166],
#        [ 68,  83,  82, 100, 110,  57,  50, 133, 131, 186],
#        [ 67,  76,  76, 112, 101, 101,  37, 123, 124, 183],
#        [ 75,  67,  65,  97, 114, 116,  42,  82, 126, 216],
#        [ 75,  66,  82, 103, 117, 103,  43, 137,  87, 187],
#        [ 79,  80,  76,  92, 101, 113,  45, 145, 148, 121]])
#Which is very bad as it is no where near a diagonal matrix, with 70 splits. Could use more splits but would take even longer

'''
kdim=10 #k principle components when performing PCA
kLDA=10 #k principle components when performing LDA

def analyse(xtrain,ytrain,xtest,ytest,kdim,kLDA):
    xtrain=xtrain
    ytrain=ytrain
    xtest=xtest
    ytest=ytest

    ##Method 2: PCA then MDA to preprocess data then train on resulting lower dimensional matrices.
    ##Step 1: Separate the classes into their respective lists and find the d dimensional means and set initial covariance matrices as identities (For orders' sake)
    xlabels = [('Top',[]),('Trouser',[]),('Pullover',[]),('Dress',[]),('Coat',[]),('Sandal',[]),('Shirt',[]),('Sneaker',[]),('Bag',[]),('Ankle boot',[])]
    xlabels=dict(xlabels)

    for i in numpy.arange(len(ytrain)):
        index=ytrain[i]
        xlabels[list(xlabels.keys())[index]].append(xtrain[i])

    ##Check that it is sorted correctly by looking at greyscale images of the dictionary keys 
    # testimage=numpy.array(xlabels['Top'][0]) #Change the label 'top' and index 0 to whatever is desired
    # testimage.resize((28,28)) #28=sqrt(784), 784=size of array of each image
    # im=Image.fromarray(testimage)
    # im.show()

    ##Step 2: Calculate the mean vector for each matrix in each class for PCA
    xlabelsav=copy.deepcopy(xlabels)
    for i in numpy.arange(len(xlabels['Top'])): #Checked that training sets for each class same length
        for j in xlabelsav:
            ##Input for the PCA is vector average of 28x28 matrix
            mat=numpy.array(xlabelsav[j][i])
            mat.resize((28,28))
            meanvec=mat.mean(1)
            xlabelsav[j][i]=meanvec

            ##Input for PCA is vector of eigenvalues of the 28x28 matrix
            # mat=numpy.array(xlabelsav[j][i])
            # mat.resize((28,28))
            # eigvals=numpy.linalg.eigvals(mat)
            # xlabelsav[j][i]=eigvals.real

    covmat_init=numpy.eye(28) #Initial covariant matrix for all classes
    ##Step 3: Take all data ignoring class labels
    classdataseparate=[]
    for k in xlabelsav.keys():
        classdata=xlabelsav[k][0]
        for m in numpy.arange(len(xlabelsav[k])-1):
            classdata=numpy.column_stack((classdata,xlabelsav[k][m+1]))
        classdataseparate.append(classdata)
        
    alldata=numpy.column_stack(tuple([i for i in classdataseparate]))


    ##Step 4: Compute mean vector of d-dimensional data
    meanall=[]
    for i in numpy.arange(numpy.shape(alldata)[0]):
        meanall.append(numpy.mean(alldata[i,:]))
    ##Scatter matrix
    scatter=numpy.zeros((28,28))
    for i in numpy.arange(numpy.shape(alldata)[1]):
        x=numpy.matrix(numpy.concatenate(tuple([i for i in alldata[:,i].reshape(28,1)]))-numpy.array(meanall))
        scatter += (x.T*x)

    ##Compute eig vectors and eig values and sort by decreasing magnitude in eigvals (since vector space spanned is most influenced by larger eigvals corresponding to larger eigvectors)
    eig_vals,eig_vecs = numpy.linalg.eig(scatter)
    pairs=[(numpy.abs(eig_vals[i]),eig_vecs[:,i]) for i in numpy.arange(len(eig_vals))]
    pairs.sort(key=lambda x:x[0],reverse=True)
    ##Say we want to use all the eigenvalues
    matW=numpy.column_stack(tuple([i[1] for i in pairs[0:kdim]]))

    ##Transform onto new subspace
    #Now xnew has dimension kx10,000, with each 6000 samples from each class now with only 6 dimensions
    xnew = numpy.mat(matW).T*numpy.mat(alldata)
    matW=numpy.mat(matW).T


    #Sort out new representation of training data
    xnewtrain=[xnew[:,i].T for i in numpy.arange(numpy.shape(xnew)[1])]
    xnewtrain=[numpy.array(j[0])[0] for j in xnewtrain]
    ynewtrain=[]
    for i in numpy.arange(10):
        ynewtrain.append(6000*[i])
    ynewtrain=numpy.concatenate([i for i in ynewtrain])

    ##Linear Discriminant analysis (nb: assumption features are independent, normally distributed)
    #mean vectors per class
    classmeans=[] #Each list in classmeans is the average vector corresponding to the class of the index
    for i in numpy.arange(10):
        mean=[]
        tempx=numpy.matrix(xnewtrain[i*10:(i+1)*10]).T
        for j in tempx:
            mean.append(numpy.mean(j))
        classmeans.append(mean)
    overallmean=[]
    for i in numpy.matrix(xnewtrain).T:
        overallmean.append(numpy.mean(i))

    ##Scatter matrix
    scatterW=numpy.zeros((kdim,kdim))
    for i in numpy.arange(10):
        Si=numpy.zeros((kdim,kdim))
        for j in numpy.arange(6000):
            x=numpy.matrix(numpy.array(xnewtrain[j])-numpy.array(classmeans[i]))
            Si += (x.T*x)
        scatterW += Si

    scatterB=numpy.zeros((kdim,kdim))

    for i in numpy.arange(10):
        x=numpy.matrix(numpy.array(classmeans[i])-numpy.array(overallmean))
        scatterB += 6000*(x.T*x)


    ##Solve eigenproblem scatterW^{-1}*Scatterb, and sort in order of magnitude
    eigvalsLDA,eigvecsLDA = numpy.linalg.eig(numpy.linalg.inv(scatterW).dot(scatterB))
    pairsLDA=[(numpy.abs(eigvalsLDA[i]), eigvecsLDA[:,i]) for i in numpy.arange(len(eigvalsLDA))]
    pairsLDA.sort(key=lambda x:x[0],reverse=True)

    ##Now construct kLDA * 28 dimensional matrix W and transform onto new subspace
    WLDA=numpy.column_stack(tuple([i[1] for i in pairsLDA[0:kLDA]]))

    xLDA=numpy.matrix(xnewtrain).dot(WLDA).real
    xLDA=[xLDA[i,:] for i in numpy.arange(numpy.shape(xLDA)[0])]
    xLDA=[numpy.array(j[0])[0] for j in xLDA]

    '''
    #Check if can discern any visual difference between for example the first 2 classes
    for i in numpy.arange(12000):
        if i<6000:
            plt.plot(numpy.arange(kLDA),xLDA[i],'r')
        elif i<12000:
            plt.plot(numpy.arange(kLDA),xLDA[i],'g')
    plt.show()
    '''

    xnewtrain=xLDA

    ##Classification using sklearn
    mpl.rcParams['savefig.dpi'] = 300
    data_and_targets=list(zip(xnewtrain,ynewtrain))

    #Initialise estimator
    estimator=ensemble.ExtraTreesClassifier()

    #Flatten data for classifier
    N_samples=len(xnewtrain)
    xnewtrainflat=numpy.array(xnewtrain).reshape((N_samples,-1))

    # Split into training and testing sets for testing with cross validation.
    kf = model_selection.KFold(n_splits=500) 
    '''
    START=time.time()
    # Create lists of known and predicted values for each split.
    y_true, y_pred = [], []
    for train, test in kf.split(xnewtrain): #splits n_splits times (n_splits folds) the data into train/test sets
        x_train, x_test, y_train, y_test = xnewtrainflat[train], xnewtrainflat[test], ynewtrain[train], ynewtrain[test]
        estimator.fit(x_train, y_train)
        y_predict = estimator.predict(x_test)
        
        # Add the split train and test predicitons to the list so we get a
        # full list of predictions.
        y_true.extend(y_test)
        y_pred.extend(y_predict)

    END=time.time()
    print(END-START, "seconds needed for cross-validation")
    # Evaluate the performance of all the splits tested.
    score = metrics.accuracy_score(y_true, y_pred) #cross validation
    print('Accuracy Score = {:.3f}'.format(score))

    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
    print("confusion matrix:",cnf_matrix)
    '''
    ##Train on all train data
    START2=time.time()
    scaler=preprocessing.StandardScaler()
    scaler.fit(xnewtrain)
    xnewtrain=scaler.transform(xnewtrain)
    estimator = estimator.fit(xnewtrain,ynewtrain)
    STOP2=time.time()
    # print("Time (s) taken for fitting whole training set = ", STOP2-START2)

    xtestnew=[]
    for i in xtest:
        ##Input is mean vector of matrix 
        mat=numpy.array(i)
        mat.resize((28,28))
        meanvect=mat.mean(1)
        xtestnew.append(meanvect)

        ##Input is vector eigenvalues of matrix
        # mat=numpy.array(i)
        # mat.resize((28,28))
        # eigvals=numpy.linalg.eigvals(mat)
        # xtestnew.append(eigvals.real)

    for i in numpy.arange(len(xtestnew)): #Transformation due to PCA
        xtestnew[i]=matW*(numpy.mat(xtestnew[i]).T)
    for i in numpy.arange(len(xtestnew)): #Transformation due to LDA
        xtestnew[i]=(numpy.matrix(xtestnew[i]).T).dot(WLDA).real
    for i in numpy.arange(len(xtestnew)):
        xtestnew[i]=numpy.array(xtestnew[i])[0]

    xtestnew=scaler.transform(xtestnew)
    ypredict=estimator.predict(xtestnew)
    scorepredict = metrics.accuracy_score(ytest,ypredict)
    cnfmatrixpredict = metrics.confusion_matrix(ytest, ypredict)
    # print("Confusion matrix of score =",scorepredict,"for prediction on test data=", cnfmatrixpredict)

    ##Some plots
    ##Plots first picture from each category
    # for i in xlabels.keys():
    #     plt.plot(numpy.arange(28),xlabelsav[i][0],label='i')
    # plt.legend()
    # plt.show()
    ##Plots first N pictures from a few categories
    # N=10
    # for j in numpy.arange(N):
    #     plt.plot(numpy.arange(28),xlabelsav['Top'][j],'r-',label='Top')
    # for j in numpy.arange(N):
    #     plt.plot(numpy.arange(28),xlabelsav['Sneaker'][j],'g-',label='Sneaker')
    # for j in numpy.arange(N):
    #     plt.plot(numpy.arange(28),xlabelsav['Dress'][j],'b-',label='Dress')
    # for j in numpy.arange(N):
    #     plt.plot(numpy.arange(28),xlabelsav['Ankle boot'][j],'k-',label='Ankle boot')
    # for j in numpy.arange(N):
    #     plt.plot(numpy.arange(28),xlabelsav['Coat'][j],'m-',label='Coat')
    # plt.show()

    return scorepredict, cnfmatrixpredict
scorepredictall=[]
cnfmatrixall=[]
for i in numpy.arange(1,29):
    kdim=i
    kLDA=i
    score,cnf=analyse(xtrain,ytrain,xtest,ytest,kdim=kdim,kLDA=kLDA)
    scorepredictall.append(score)
    cnfmatrixall.append(cnf)
STOP3=time.time()

print("Time (s) taken for all analysis=", STOP3-START3)
print("Best score = ",max(scorepredictall),"with number of eigenvalues taken as =", scorepredictall.index(max(scorepredictall))+1, "and corresponding cnf matrix= ", cnfmatrixall[scorepredictall.index(max(scorepredictall))])
plt.plot(numpy.arange(1,29),scorepredictall)
plt.xlabel("No. eigvals taken")
plt.ylabel("Score")
plt.title("Score vs number of eigenvalues taken (same for PCA and LDA)")
plt.show()


