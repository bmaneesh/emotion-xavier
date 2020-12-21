# VR with CNN EEG features
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import scipy.io as sio
from keras.layers import Conv1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks  import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,roc_auc_score, roc_curve, auc
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD,adam
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

dataPath = './data/'
savePath = './Results/'
# experiments = ['Gender_Clean', 'Gender_notClean','Emotion_Clean', 'Emotion_notClean'] #  
# experiments = ['Emotion_Clean', 'Emotion_notClean'] # 
experiments = ['Emotion_notClean'] # 
classifier = 'CNN' #'Both' CNN' 'SVM'
foldNum = 10

# Setting CNN Parameters
n_cpu = -2 #SVM parallel processes
nb_filters = [16, 32, 32, 64]
kernel_size = 3
pool_size = 2
stride_size = 2
nb_epoch = [10, 10, 10]  
learn_rate = 0.01
batch_size = 32
mtm = 0.9
padding = 'same'
dense_layer_neuron_num = 128
dropout_level = 0.1
weight_decay = 0.000001

for experiment in experiments:
    # Loading data
    if (experiment == 'Gender_Clean' or experiment == 'Emotion_Clean'):
        matContent = sio.loadmat(dataPath + 'all_data_clean.mat')
    elif (experiment == 'Gender_notClean' or experiment == 'Emotion_notClean'):
        matContent = sio.loadmat(dataPath + 'all_data_notclean.mat')
    features = matContent['features']
    if (experiment == 'Gender_Clean' or experiment == 'Gender_notClean'):
        labels = np.squeeze(matContent['labels_gender'])
        f1Net = np.zeros([foldNum,])
        precisionNet = np.zeros([foldNum,])
        recallNet = np.zeros([foldNum,])
        accNet = np.zeros([foldNum,])
        aucNet = np.zeros([foldNum,])           #comment AUC metric lines for org. ER code
        f1SVM = np.zeros([foldNum,])
        precisionSVM = np.zeros([foldNum,])
        recallSVM = np.zeros([foldNum,])
        accSVM = np.zeros([foldNum,])
        aucSVM = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
    elif (experiment == 'Emotion_Clean' or experiment == 'Emotion_notClean'):
        labels = np.squeeze(matContent['labels_emotion']) - 1 
        accNet = np.zeros([foldNum,])
        aucNet = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
        accSVM = np.zeros([foldNum,])
        aucSVM = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
    del matContent
    channels = features.shape[2]
    #randomise the sample sequence
    rand_order = np.arange(features.shape[0])
    np.random.shuffle(rand_order)
    features = features[rand_order,]
    labels = np.squeeze(labels[rand_order,])
    #####
    if (experiment == 'Emotion_Clean' or experiment == 'Emotion_notClean'):# valence label modification
        # print np.where(labels==0)[0].tolist() + np.where(labels==1)[0].tolist()
        labels[np.where(labels==0)[0].tolist() + np.where(labels==1)[0].tolist() + np.where(labels==2)[0].tolist() + np.where(labels==4)[0].tolist(),] = 0
        labels[np.where(labels == 3)[0].tolist() + np.where(labels == 5)[0].tolist(),] = 1
    class_num = np.size(np.unique(labels))
    labels_categorical = np_utils.to_categorical(labels, class_num)    
    skf = StratifiedKFold(n_splits = foldNum)
    f = 0
    for train, test in skf.split(features, labels):
        # Filling the missing values and Normalization
        print('Data Prepration ... \n')
        trainingFeatures = features[train,:,:]
        testFeatures = features[test,:,:]
        train_shape = trainingFeatures.shape
        test_shape = testFeatures.shape
        imputer = Imputer()
        imputer.fit(np.reshape(trainingFeatures, [train_shape[0], train_shape[1]*train_shape[2]]))
        trainingFeatures = imputer.transform(np.reshape(trainingFeatures, [train_shape[0], train_shape[1]*train_shape[2]]))
        testFeatures = imputer.transform(np.reshape(testFeatures, [test_shape[0], test_shape[1]*test_shape[2]]))
        
        scaler = StandardScaler()
        scaler.fit(trainingFeatures)
        trainingFeatures = scaler.transform(trainingFeatures)
        trainingFeatures = np.reshape(trainingFeatures, [train_shape[0],train_shape[1],train_shape[2]])
        testFeatures = scaler.transform(testFeatures)
        testFeatures = np.reshape(testFeatures,[test_shape[0], test_shape[1], test_shape[2]])
        # Training and evaluations
        if (classifier == 'CNN' or classifier == 'Both'):
            # CNN
            print('Training the CNN Model ... \n')
            model = Sequential()
            model.add(Conv1D(filters=nb_filters[0], kernel_size=kernel_size, padding=padding, activation='relu',
                         input_shape=(train_shape[1], train_shape[2]), trainable=False))
            model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
            model.add(Conv1D(filters=nb_filters[1], kernel_size=kernel_size, padding=padding, activation='relu',
                         kernel_initializer='he_normal', trainable=False))
            model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
            model.add(Conv1D(filters=nb_filters[2], kernel_size=kernel_size, padding=padding, activation='relu',
                         kernel_initializer='he_normal', trainable=False))
            model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
            # ####added by me#####
            # model.add(Conv1D(filters=nb_filters[2], kernel_size=kernel_size, padding=padding, activation='relu',
            #              kernel_initializer='he_normal'))
            # model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
            # ####added by me#####
            model.add(Flatten())
            model.add(BatchNormalization(epsilon=0.001))
            model.add(Dense(dense_layer_neuron_num, kernel_initializer='he_normal', activation='relu'))
            model.add(Dropout(dropout_level))
            model.add(Dense(class_num))
            model.add(Activation('softmax'))
            # model.summary()
            model.load_weights('Gender_notClean_HIweights.hdf5')
            earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
            for ep in range(len(nb_epoch)):
                sgd = SGD(lr=learn_rate / 10 ** ep, momentum = mtm, decay = weight_decay, nesterov = True)
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
                model.fit(trainingFeatures, labels_categorical[train,:], batch_size=trainingFeatures.shape[0], epochs=nb_epoch[ep],
                          verbose=2, callbacks=[earlyStopping], validation_split=0.1) 
            # sgd = adam()
            # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            # model.fit(trainingFeatures, labels_categorical[train,:], batch_size=batch_size, epochs=nb_epoch,
            #           verbose=2, callbacks=[earlyStopping], validation_split=0.1) 
            predicted_labelsNet = model.predict_classes(testFeatures, verbose=0)
            predicted_probsNet = model.predict_proba(testFeatures,batch_size=1,verbose=0)
            if (experiment == 'Gender_Clean' or experiment == 'Gender_notClean'):
                precisionNet[f] = precision_score(labels[test,], predicted_labelsNet, average='macro')
                recallNet[f] = recall_score(labels[test,], predicted_labelsNet, average='macro')
                f1Net[f] = f1_score(labels[test,], predicted_labelsNet, average='macro')
                accNet[f] = accuracy_score(labels[test,], predicted_labelsNet)
                aucNet[f] = roc_auc_score(labels[test,], predicted_probsNet[:,1])
                print(experiment + '_CNN: Fold %d : f1_score: %.4f' % (f + 1, f1Net[f]))
                print(experiment + '_CNN: Fold %d : auc: %.4f' % (f + 1, aucNet[f]))
            elif (experiment == 'Emotion_Clean' or experiment == 'Emotion_notClean'):
                accNet[f] = accuracy_score(labels[test,], predicted_labelsNet)
                aucNet[f] = roc_auc_score(labels[test,], predicted_probsNet[:,1])
                print(experiment + '_CNN: Fold %d : acc: %.4f' % (f + 1, accNet[f]))
                print(experiment + '_CNN: Fold %d : auc: %.4f' % (f + 1, aucNet[f]))
        if (classifier == 'SVM' or classifier == 'Both'):
            # SVM
            print('Training the SVM Model ... \n')
            if (experiment == 'Gender_Clean' or experiment == 'Gender_notClean'):
                parameters = {'kernel':('linear', 'rbf'), 'C':[0.01,0.1,1.0,10.0]}
                svc = SVC(probability=True)
                clf = GridSearchCV(svc, parameters, cv = 2, verbose = 2, n_jobs = n_cpu)
                clf.fit(np.reshape(trainingFeatures, [train_shape[0], train_shape[1]*train_shape[2]]), labels[train,]) 
                predicted_labelsSVM = clf.predict(np.reshape(testFeatures, [test_shape[0], test_shape[1]*test_shape[2]]))
                predicted_probsSVM = clf.predict_proba(np.reshape(testFeatures, [test_shape[0], test_shape[1]*test_shape[2]]))
                precisionSVM[f] = precision_score(labels[test,], predicted_labelsSVM)
                recallSVM[f] = recall_score(labels[test,], predicted_labelsSVM)
                f1SVM[f] = f1_score(labels[test,], predicted_labelsSVM, average='macro')
                accSVM[f] = accuracy_score(labels[test,], predicted_labelsSVM)
                aucSVM[f] = roc_auc_score(labels[test,], predicted_probsSVM[:,1])
                print(experiment + '_SVM: Fold %d : f1_score: %.4f' % (f + 1, f1SVM[f])) 
                print(experiment + '_SVM: Fold %d : auc_score: %.4f' % (f + 1, aucSVM[f])) 
            elif (experiment == 'Emotion_Clean' or experiment == 'Emotion_notClean'):
                parameters = {'estimator__C':[0.01,0.1,1.0,10.0]}
                svc = OneVsRestClassifier(LinearSVC(),n_jobs=n_cpu)
                clf = CalibratedClassifierCV(GridSearchCV(svc, parameters, cv = 2,  verbose = 2),cv = 2)
                # clf = OneVsRestClassifier(CalibratedClassifierCV(GridSearchCV(LinearSVC(), parameters, cv = 2,  verbose = 2),cv = 2),n_jobs=4)
                # clf.fit(np.reshape(trainingFeatures, [train_shape[0], train_shape[1]*train_shape[2]]), labels[train,]) 
                clf.fit(np.reshape(trainingFeatures, [train_shape[0], train_shape[1]*train_shape[2]]), labels[train,])
                predicted_labelsSVM = clf.predict(np.reshape(testFeatures, [test_shape[0], test_shape[1]*test_shape[2]]))
                predicted_probsSVM = clf.predict_proba(np.reshape(testFeatures, [test_shape[0], test_shape[1]*test_shape[2]]))
                # predicted_probsSVM = clf.decision_function(np.reshape(testFeatures, [testFeatures.shape[0], testFeatures.shape[1]*testFeatures.shape[2]]))
                accSVM[f] = accuracy_score(labels[test,], predicted_labelsSVM)
                aucSVM[f] = roc_auc_score(labels[test,], predicted_probsSVM[:,1])
                print(experiment + '_SVM: Fold %d : acc: %.4f' % (f + 1, accSVM[f]))
                print(experiment + '_SVM: Fold %d : auc: %.4f' % (f + 1, aucSVM[f]))
        f += 1
    # Saving the results
    if (classifier == 'Both'):
        if (experiment == 'Gender_Clean' or experiment == 'Gender_notClean'):
            sio.savemat(savePath + experiment + '_Results_CNN_VR' + '.mat', {'precisionNet': precisionNet,'recallNet': recallNet, 'f1Net': f1Net, 'accNet': accNet,'aucNet': aucNet,
                                                    'precisionSVM': precisionSVM,'recallSVM': recallSVM, 'f1SVM': f1SVM, 'accSVM': accSVM, 'aucSVM': aucSVM})
        elif (experiment == 'Emotion_Clean' or experiment == 'Emotion_notClean'):
            sio.savemat(savePath + experiment + classifier + '_Results_CNN_VR' + '.mat', {'accNet': accNet, 'accSVM': accSVM, 'accNet': accNet, 'aucNet': aucNet, 'aucSVM': aucSVM})
    elif (classifier == 'CNN'):
        if (experiment == 'Gender_Clean' or experiment == 'Gender_notClean'):
            sio.savemat(savePath + experiment + '_Results_CNN_VR' + '.mat', {'precisionNet': precisionNet,'recallNet': recallNet, 'f1Net': f1Net, 'accNet': accNet, 'aucNet': aucNet})
        elif (experiment == 'Emotion_Clean' or experiment == 'Emotion_notClean'):
            sio.savemat(savePath + experiment + classifier + '_Results_CNN_VR' + '.mat', {'accNet': accNet, 'aucNet': aucNet})
    elif (classifier == 'SVM'):
        if (experiment == 'Gender_Clean' or experiment == 'Gender_notClean'):
            sio.savemat(savePath + experiment + '_Results__CNN_VR' + '.mat', {'precisionSVM': precisionSVM,'recallSVM': recallSVM, 'f1SVM': f1SVM, 'accSVM': accSVM, 'aucSVM': aucSVM})
        elif (experiment == 'Emotion_Clean' or experiment == 'Emotion_notClean'):
            sio.savemat(savePath + experiment + classifier + '_Results_CNN_VR' + '.mat', {'accSVM': accSVM, 'aucNet': aucNet})
