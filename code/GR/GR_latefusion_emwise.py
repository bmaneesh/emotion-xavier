# Emotionwise GR late fusion with CNN and AdaBoost for HI/LI/EYE/MOUTH mask conditions
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import scipy.io as sio
from keras.layers import Conv1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks  import EarlyStopping, ModelCheckpoint
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
from sklearn.ensemble import AdaBoostClassifier
import math

dataPath = '/home/maneesh/journal_fusion/gaze_based_eeg_selected/'
savePath = '/home/maneesh/Downloads/DNN/Results/'
experiments = ['Gender_notClean_HI', 'Gender_notClean_LI', 'Gender_notClean_EYE', 'Gender_notClean_MOUTH'] #  
# experiments = [ 'Gender_notClean_EYE', 'Gender_notClean_MOUTH'] #  ]
classifier = 'CNN' #'Both' CNN' 'SVM'
foldNum = 10
emotion = ['Anger','Disgust','Fear','Happy','Sad','Surprise']

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
n_est = 10
input_feat_size = 128
class_num = 2

f1Net = {}
precisionNet = {}
recallNet = {}
accNet = {}
aucNet = {}
accAda = {}
aucAda = {}
f1Ada = {}
precisionAda = {}
recallAda = {}
accBoth = {}
aucBoth = {}
f1Both = {}
skf = StratifiedKFold(foldNum)

for experiment in experiments:
    f1Net[experiment] = np.zeros([foldNum,])
    precisionNet[experiment] = np.zeros([foldNum,])
    recallNet[experiment] = np.zeros([foldNum,])
    accNet[experiment] = np.zeros([foldNum,])
    aucNet[experiment] = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
    accAda[experiment] = np.zeros([foldNum,])
    aucAda[experiment] = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
    f1Ada[experiment] = np.zeros([foldNum,])
    precisionAda[experiment] = np.zeros([foldNum,])
    recallAda[experiment] = np.zeros([foldNum,])
    accBoth[experiment] = np.zeros([foldNum,])
    aucBoth[experiment] = np.zeros([foldNum,])
    f1Both[experiment] = np.zeros([foldNum,])

def data_load_casebasis(experiment_name):
    split_name = experiment_name.split('_')
    if split_name[2] == 'EYE' or experiment_name.split('_')[2] == 'MOUTH':
        matContent = sio.loadmat(dataPath+'results2/journal_fuse_results2')
        features_eeg = matContent['features_eeg']
        features_gaze = matContent['features_gaze']
        labels = matContent['gender']
        em_label = matContent['uresp_mat'][:,1]
        if split_name[2]=='EYE':
            idx = np.intersect1d(np.where(matContent['uresp_mat'][:,4]==1),np.where(matContent['uresp_mat'][:,2] == matContent['uresp_mat'][:,1]))
        else:
            idx = np.intersect1d(np.where(matContent['uresp_mat'][:,4]==0),np.where(matContent['uresp_mat'][:,2] == matContent['uresp_mat'][:,1]))
    else:
        matContent = sio.loadmat(dataPath+'results1/journal_fuse_results1')
        features_eeg = matContent['features_eeg']
        features_gaze = matContent['features_gaze']
        labels = matContent['gender']
        em_label = matContent['uresp_mat'][:,1]
        if split_name[2]=='HI':
            idx = np.intersect1d(np.where(matContent['uresp_mat'][:,0]==1),np.where(matContent['uresp_mat'][:,2] == 1))
        else:
            idx = np.intersect1d(np.where(matContent['uresp_mat'][:,0]==0),np.where(matContent['uresp_mat'][:,2] == 1))
    features_eeg = features_eeg[:,:,idx]
    features_gaze = features_gaze[idx,:]
    labels = labels[idx,]
    em_label = em_label[idx,]
    features_eeg = np.transpose(features_eeg,[2,1,0])
    features_eeg = downsample(factor = 4,mat =features_eeg[:,65:,:],axis= 1)
    return features_eeg, features_gaze, np.squeeze(labels), np.squeeze(em_label)

def model_def(nb_filters,kernel_size,padding,input_shape,trainable,dropout_level,class_num):
    # model = Sequential()
    # model.add(Conv1D(filters=nb_filters[0], kernel_size=kernel_size, padding=padding, activation='relu',
    #              input_shape=(input_shape[0], input_shape[1])))
    # model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    # model.add(Conv1D(filters=nb_filters[1], kernel_size=kernel_size, padding=padding, activation='relu',
    #              kernel_initializer='he_normal'))
    # model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    # model.add(Conv1D(filters=nb_filters[2], kernel_size=kernel_size, padding=padding, activation='relu',
    #              kernel_initializer='he_normal'))
    # model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    # model.add(Flatten())
    # model.add(BatchNormalization(epsilon=0.001))
    # model.add(Dense(dense_layer_neuron_num, kernel_initializer='he_normal', activation='relu'))
    # model.add(Dropout(dropout_level))
    # model.add(Dense(class_num))
    # model.add(Activation('softmax'))
    model = Sequential()
    model.add(Conv1D(filters=nb_filters[0], kernel_size=kernel_size, padding=padding, activation='relu',
                 input_shape=(train_shape[1], train_shape[2])))
    model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    model.add(Conv1D(filters=nb_filters[1], kernel_size=kernel_size, padding=padding, activation='relu',
                 kernel_initializer='he_normal'))
    model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    model.add(Conv1D(filters=nb_filters[2], kernel_size=kernel_size, padding=padding, activation='relu',
                 kernel_initializer='he_normal'))
    model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    # model.add(Conv1D(filters=nb_filters[3], kernel_size=kernel_size, padding=padding, activation='relu',
    #              kernel_initializer='he_normal'))
    # model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))    
    model.add(Flatten())
    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dense(dense_layer_neuron_num, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(dropout_level))
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    return model

def downsample(factor,mat,axis):
    mat_shape = mat.shape
    new_mat = np.empty([mat.shape[0],1,mat.shape[2]])
    for i in range(0,mat_shape[axis],factor):
        new_mat = np.append(new_mat,np.reshape(mat[:,i,:],[mat.shape[0],1,mat.shape[2]]),axis=1)
    new_mat = np.delete(new_mat, 0,1)
    return new_mat


if __name__ == '__main__':
    for experiment in experiments:
        for em in range(0,len(emotion)):
            'Processing-'+emotion[em]
            features_eeg_all,features_gaze_all,labels_all,em_label_all = data_load_casebasis(experiment)
            rand_order = np.arange(features_eeg_all.shape[0])
            np.random.shuffle(rand_order)
            features_eeg_all = features_eeg_all[rand_order,]
            features_gaze_all = features_gaze_all[rand_order,]
            labels_all = np.squeeze(labels_all[rand_order,])
            em_label_all = np.squeeze(em_label_all[rand_order,])-1
            class_num = np.size(np.unique(labels_all))
            idx = np.squeeze(np.where(em_label_all==em))
            # print len(idx)
            features_eeg = features_eeg_all[idx,:,:]
            features_gaze = features_gaze_all[idx,:]
            labels = np.squeeze(labels_all[idx,])
            f = 0
            # print features_eeg.shape,labels.shape,idx.shape
            for train,test in skf.split(features_eeg,labels):
                # print features_eeg.shape
                imputer = Imputer()
                train_features = features_eeg[train,]
                test_features = features_eeg[test,]
                train_shape = features_eeg[train,].shape
                test_shape = features_eeg[test,].shape
                imputer.fit(np.reshape(train_features,[train_shape[0],train_shape[1]*train_shape[2]]))
                train_features = imputer.transform(np.reshape(train_features,[train_shape[0],train_shape[1]*train_shape[2]]))
                test_features = imputer.transform(np.reshape(test_features,[test_shape[0],test_shape[1]*test_shape[2]]))
                scalar = StandardScaler()
                scalar.fit(train_features)
                train_features = scalar.transform(train_features)
                test_features = scalar.transform(test_features)
                train_features = np.reshape(train_features,[train_shape[0],train_shape[1],train_shape[2]])
                test_features = np.reshape(test_features,[test_shape[0],test_shape[1],test_shape[2]])
                dl_model = model_def(nb_filters=nb_filters, kernel_size=kernel_size, padding=padding, input_shape=(input_feat_size,14), trainable=True, dropout_level=dropout_level, class_num=class_num)
                earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
                split_name = experiment.split('_')
                # dl_model.load_weights('Gender_notClean_'+split_name[2]+'weights.hdf5')#toggle between Clean and notClean here
                earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
            # model_save = ModelCheckpoint(experiment+'weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
                for ep in range(len(nb_epoch)):
                    sgd = SGD(lr=learn_rate / 10 ** ep, momentum = mtm, decay = weight_decay, nesterov = True)
                    dl_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                    dl_model.fit(train_features, np_utils.to_categorical(labels[train,], class_num), batch_size=batch_size, epochs=nb_epoch[ep],
                          verbose=2, callbacks=[earlyStopping], validation_split=0.1) 
                # for ep in nb_epoch:
                #     sgd = SGD(lr=learn_rate / 10 ** ep, momentum = mtm, decay = weight_decay, nesterov = True)
                #     dl_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                #     dl_model.fit(train_features, np_utils.to_categorical(labels[train,], class_num), batch_size=batch_size, epochs=ep,
                #         verbose=2,  validation_split=0.1, callbacks=[earlyStopping])
                predicted_labelsNet = dl_model.predict_classes(test_features, verbose=0)
                predicted_probsNet = dl_model.predict_proba(test_features,batch_size=1,verbose=0)
                accNet[experiment][f] = accuracy_score(labels[test,], predicted_labelsNet)
                # print np.unique(predicted_probsNet)
                aucNet[experiment][f] = roc_auc_score(labels[test,], predicted_probsNet[:,1])
                f1Net[experiment][f] = f1_score(labels[test,], predicted_labelsNet, average='macro')
                precisionNet[experiment][f] = precision_score(labels[test,], predicted_labelsNet, average='macro')
                recallNet[experiment][f] = recall_score(labels[test,], predicted_labelsNet, average='macro')
                print(experiment + '_CNN: Fold %d : acc: %.4f' % (f + 1, accNet[experiment][f]))
                print(experiment + '_CNN: Fold %d : auc: %.4f' % (f + 1, aucNet[experiment][f]))
                print(experiment + '_CNN: Fold %d : f1: %.4f' % (f + 1, f1Net[experiment][f]))
                train_features = features_gaze[train,]
                test_features = features_gaze[test,]
                imputer.fit(train_features)
                train_features = imputer.transform(train_features)
                test_features = imputer.transform(test_features)
                scalar = StandardScaler()
                scalar.fit(train_features)
                train_features = scalar.transform(train_features)
                test_features = scalar.transform(test_features)
                ada_model = AdaBoostClassifier(algorithm='SAMME',n_estimators = n_est, random_state=245641)
                ada_model.fit(train_features,labels[train,])
                predicted_labelsAda = ada_model.predict(test_features)
                predicted_probsAda = ada_model.predict_proba(test_features)
                accAda[experiment][f] = accuracy_score(labels[test,], predicted_labelsAda)
                aucAda[experiment][f] = roc_auc_score(labels[test,], predicted_probsAda[:,1])
                f1Ada[experiment][f] = f1_score(labels[test,], predicted_labelsAda, average='macro')
                precisionAda[experiment][f] = precision_score(labels[test,], predicted_labelsAda, average='macro')
                recallAda[experiment][f] = recall_score(labels[test,], predicted_labelsAda, average='macro')
                # print(experiment + '_Ada: Fold %d : acc: %.4f' % (f + 1, accAda[experiment][f]))
                # print(experiment + '_Ada: Fold %d : auc: %.4f' % (f + 1, aucAda[experiment][f]))
                # print(experiment + '_Ada: Fold %d : f1: %.4f' % (f + 1, f1Ada[experiment][f]))
                accBoth[experiment][f] = accuracy_score(labels[test,], (predicted_probsAda[:,1]+predicted_probsNet[:,1])/2>0.5)
                aucBoth[experiment][f] = roc_auc_score(labels[test,], (predicted_probsAda[:,1]+predicted_probsNet[:,1])/2)
                f1Both[experiment][f] = f1_score(labels[test,], (predicted_probsAda[:,1]+predicted_probsNet[:,1])/2>0.5,average='macro')
                f+=1
            # print(experiment + '_CNN: Mean %.4f : acc: %.4f' % (np.mean(accNet[experiment]),np.std(accNet[experiment])))
            print(experiment+emotion[em] + '_CNN: Mean %.4f : auc: %.4f' % (np.mean(aucNet[experiment]),np.std(aucNet[experiment])))
            # print(experiment + '_CNN: Mean %.4f : f1: %.4f' % (np.mean(f1Net[experiment]),np.std(f1Net[experiment])))

            # print(experiment + '_Ada: Mean %.4f : acc: %.4f' % (np.mean(accAda[experiment]),np.std(accAda[experiment])))
            print(experiment+emotion[em] + '_Ada: Mean %.4f : auc: %.4f' % (np.mean(aucAda[experiment]),np.std(aucAda[experiment])))
            # print(experiment + '_Ada: Mean %.4f : f1: %.4f' % (np.mean(f1Ada[experiment]),np.std(f1Ada[experiment])))
            print(experiment+emotion[em] + '_Both: Mean %.4f : auc: %.4f' % (np.mean(aucBoth[experiment]),np.std(aucBoth[experiment])))
            sio.savemat(experiment+emotion[em]+'_fuse.mat', {'accNet': accNet[experiment], 'accAda': accAda[experiment], 'accBoth': accBoth[experiment], 'aucNet': accNet[experiment], 'aucAda': aucAda[experiment], 'aucBoth': aucBoth[experiment]
                , 'f1Net': f1Net[experiment], 'f1Ada': f1Ada[experiment], 'f1Both': f1Both[experiment]})