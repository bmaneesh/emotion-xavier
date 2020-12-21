# VR with CNN with EEG features for HI/LI/EYE/MOUTH mask conditions
import os
import csv
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import scipy.io as sio
from keras.layers import Conv1D, AveragePooling1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks  import EarlyStopping, ReduceLROnPlateau
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

dataPath = './../DNN/Data/'
savePath = '/home/maneesh/Downloads/DNN/Results/'
# experiments = ['Gender_Clean', 'Gender_notClean','Emotion_Clean', 'Emotion_notClean'] #  
# experiments = ['Emotion_Clean_HI']#, 'Emotion_notClean'] # 
experiments = ['Emotion_Clean_HI','Emotion_notClean_HI','Emotion_Clean_LI','Emotion_notClean_LI','Emotion_Clean_EYE','Emotion_notClean_EYE','Emotion_Clean_MOUTH','Emotion_notClean_MOUTH'] # 
# experiments = ['Emotion_notClean_HI'] # 
classifier = 'CNN' #'Both' CNN' 'SVM'
foldNum = 10
gen_wise = 1

# best__Emotion_notClean_maleCNN_Results_VR_0.00176091259056_16_0.1_0.000130102999566
# Setting CNN Parameters
n_cpu = -2 #SVM parallel processes
nb_filters = [16, 32, 32, 64]
kernel_size = 3
pool_size = 2
stride_size = 2
nb_epoch = [10, 10, 10]  
learn_rate = 0.0017609
batch_size = 16
mtm = 0.9
padding = 'same'
dense_layer_neuron_num = 128
dropout_level = 0.1
weight_decay = 0.0001

with open('VR_log.csv', 'a') as csvfile:
    fieldnames = ['Experiment','precisionNet', 'recallNet','f1Net','accNet','aucNet']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for gender in [0,1]:
    for experiment in experiments:
        # Loading data
        if (experiment == 'Emotion_Clean_HI'):
            matContent = sio.loadmat(dataPath + 'eeg_hi_cor.mat')
            features = matContent['hi_mat']
            labels = np.squeeze(matContent['er_hi_label'])-1
            features = features[np.squeeze(np.where(np.squeeze(matContent['gr_hi_label']==gender)))]
            labels = labels[np.squeeze(np.where(np.squeeze(matContent['gr_hi_label']==gender)))];
        elif (experiment == 'Emotion_notClean_HI'):
            matContent = sio.loadmat(dataPath + 'eeg_hi_cor_notClean.mat')
            features = matContent['hi_mat']
            labels = np.squeeze(matContent['er_hi_label'])-1      
            features = features[np.squeeze(np.where(np.squeeze(matContent['gr_hi_label']==gender)))]
            labels = labels[np.squeeze(np.where(np.squeeze(matContent['gr_hi_label']==gender)))];
        elif (experiment == 'Emotion_Clean_LI'):
            matContent = sio.loadmat(dataPath + 'eeg_li_cor.mat')
            features = matContent['li_mat']
            labels = np.squeeze(matContent['er_li_label'])-1
            features = features[np.squeeze(np.where(np.squeeze(matContent['gr_li_label']==gender)))]
            labels = labels[np.squeeze(np.where(np.squeeze(matContent['gr_li_label']==gender)))];
        elif (experiment == 'Emotion_notClean_LI'):
            matContent = sio.loadmat(dataPath + 'eeg_li_cor_notClean.mat')
            features = matContent['li_mat']
            labels = np.squeeze(matContent['er_li_label'])-1
            features = features[np.squeeze(np.where(np.squeeze(matContent['gr_li_label']==gender)))]
            labels = labels[np.squeeze(np.where(np.squeeze(matContent['gr_li_label']==gender)))];
        elif (experiment == 'Emotion_Clean_EYE'):
            matContent = sio.loadmat(dataPath + 'eeg_eye_hi_cor.mat')
            features = matContent['eye_mat']
            labels = np.squeeze(matContent['er_eye_label'])-1
            features = features[np.squeeze(np.where(np.squeeze(matContent['gr_eye_label']==gender)))]
            labels = labels[np.squeeze(np.where(np.squeeze(matContent['gr_eye_label']==gender)))];
        elif (experiment == 'Emotion_notClean_EYE'):
            matContent = sio.loadmat(dataPath + 'eeg_eye_hi_cor_notClean.mat')
            features = matContent['eye_mat']
            labels = np.squeeze(matContent['er_eye_label'])-1
            features = features[np.squeeze(np.where(np.squeeze(matContent['gr_eye_label']==gender)))]
            labels = labels[np.squeeze(np.where(np.squeeze(matContent['gr_eye_label']==gender)))];
        elif (experiment == 'Emotion_Clean_MOUTH'):
            matContent = sio.loadmat(dataPath + 'eeg_mouth_hi_cor.mat')
            features = matContent['mouth_mat']
            labels = np.squeeze(matContent['er_mouth_label'])-1
            features = features[np.squeeze(np.where(np.squeeze(matContent['gr_mouth_label']==gender)))]
            labels = labels[np.squeeze(np.where(np.squeeze(matContent['gr_mouth_label']==gender)))];
        elif (experiment == 'Emotion_notClean_MOUTH'):
            matContent = sio.loadmat(dataPath + 'eeg_mouth_hi_cor_notClean.mat')
            features = matContent['mouth_mat']
            labels = np.squeeze(matContent['er_mouth_label'])-1
            features = features[np.squeeze(np.where(np.squeeze(matContent['gr_mouth_label']==gender)))]
            labels = labels[np.squeeze(np.where(np.squeeze(matContent['gr_mouth_label']==gender)))];
        # features = matContent['features']
        if gen_wise:
            labels[np.where(labels==0)[0].tolist() + np.where(labels==1)[0].tolist() + np.where(labels==2)[0].tolist() + np.where(labels==4)[0].tolist(),] = 0
            labels[np.where(labels == 3)[0].tolist() + np.where(labels == 5)[0].tolist(),] = 1
            f1Net = np.zeros([foldNum,])
            precisionNet = np.zeros([foldNum,])
            recallNet = np.zeros([foldNum,])
            accNet = np.zeros([foldNum,])
            aucNet = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
            accSVM = np.zeros([foldNum,])
            aucSVM = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
            # del matContent
            channels = features.shape[2]
            #randomise the sample sequence
            rand_order = np.arange(features.shape[0])
            np.random.shuffle(rand_order)
            features = features[rand_order,]
            labels = np.squeeze(labels[rand_order,])
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
                                 kernel_initializer='he_normal', trainable=True))
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
                    model.load_weights('Gender_'+experiment[8:]+'weights.hdf5')
                    # print 'Successfully loaded '+'Gender_'+experiment[8:]+'weights.hdf5'
                    # model.summary()
                    earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
                    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
                    for ep in range(len(nb_epoch)):
                        sgd = SGD(lr=learn_rate / 10 ** ep, momentum = mtm, decay = weight_decay, nesterov = True)
                        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
                        model.fit(trainingFeatures, labels_categorical[train,:], batch_size=batch_size, epochs=nb_epoch[ep],
                                  verbose=2,  validation_split=0.1, callbacks=[earlyStopping])
                    # sgd = adam()#SGD(lr=learn_rate, momentum = mtm, decay = weight_decay, nesterov = True)
                    # model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
                    # model.fit(trainingFeatures, labels_categorical[train,:], batch_size=batch_size, epochs=50,
                    #           verbose=2,  validation_split=0.2, callbacks=[earlyStopping])
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
                    elif (experiment[0][0] == 'E'):
                        accNet[f] = accuracy_score(labels[test,], predicted_labelsNet)
                        aucNet[f] = roc_auc_score(labels[test,], predicted_probsNet[:,1])
                        f1Net[f] = f1_score(labels[test,], predicted_labelsNet, average='macro')
                        precisionNet[f] = precision_score(labels[test,], predicted_labelsNet, average='macro')
                        recallNet[f] = recall_score(labels[test,], predicted_labelsNet, average='macro')
                        print(experiment + '_CNN: Fold %d : acc: %.4f' % (f + 1, accNet[f]))
                        print(experiment + '_CNN: Fold %d : auc: %.4f' % (f + 1, aucNet[f]))
                        print(experiment + '_CNN: Fold %d : f1: %.4f' % (f + 1, f1Net[f]))
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
        if gender == 0:                                                                     #gender=1 female
            experiment = experiment + '_male'
        else:
            experiment = experiment + '_female'
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
            elif (experiment[0][0] == 'E'):
                with open('VR_log.csv', 'a') as csvfile:
                    # fieldnames = ['Experiment','precisionNet', 'recallNet','f1Net','accNet','aucNet']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    # writer.writeheader()
                    writer.writerow({'Experiment':experiment,'precisionNet': str(np.mean(precisionNet))+'+-'+str(np.std(precisionNet)),'recallNet': str(np.mean(recallNet))+'+-'+str(np.std(recallNet)),
                    'f1Net': str(np.mean(f1Net))+'+-'+str(np.std(f1Net)),'accNet': str(np.mean(accNet))+'+-'+str(np.std(accNet)), 'aucNet': str(np.mean(aucNet))+'+-'+str(np.std(aucNet))})
                sio.savemat(savePath + experiment + classifier + '_Results_VR' + '.mat', {'precisionNet': precisionNet,'recallNet': recallNet, 'f1Net': f1Net,'accNet': accNet, 'aucNet': aucNet})
        elif (classifier == 'SVM'):
            if (experiment == 'Gender_Clean' or experiment == 'Gender_notClean'):
                sio.savemat(savePath + experiment + '_Results__CNN_VR' + '.mat', {'precisionSVM': precisionSVM,'recallSVM': recallSVM, 'f1SVM': f1SVM, 'accSVM': accSVM, 'aucSVM': aucSVM})
            elif (experiment == 'Emotion_Clean' or experiment == 'Emotion_notClean'):
                sio.savemat(savePath + experiment + classifier + '_Results_CNN_VR' + '.mat', {'accSVM': accSVM, 'aucNet': aucNet})
