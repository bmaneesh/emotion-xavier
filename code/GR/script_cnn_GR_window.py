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

dataPath = './data/'
savePath = './Results/'
# experiments = ['Gender_notClean_HI', 'Gender_notClean_LI', 'Gender_notClean_EYE','Gender_notClean_MOUTH'] #  
experiments = [ 'Gender_notClean_MOUTH'] #  'Gender_notClean_HI','Gender_notClean_LI','Gender_notClean_EYE',
# experiments = ['Emotion_Clean', 'Emotion_notClean'] # 
classifier = 'CNN' #'Both' CNN' 'SVM'
em_list = ['anger','disgust','fear','happy','sad','surprise']
foldNum = 10
em_wise = 1
n_em = 6

# Setting CNN Parameters
n_cpu = -2 #SVM parallel processes
nb_filters = [16, 32, 32, 64]
kernel_size = 3
pool_size = 2
stride_size = 2
nb_epoch = [20, 20, 20]  
learn_rate = 0.01
batch_size = 32
mtm = 0.9
padding = 'same'
dense_layer_neuron_num = 128
dropout_level = 0.1
weight_decay = 0.000001

# nb_filters = [16, 32, 32, 64]
# kernel_size = 3
# pool_size = 2
# stride_size = 2
# nb_epoch = [20, 20, 20]  
# learn_rate = 0.0017
# batch_size = 32
# mtm = 0.9
# padding = 'same'
# dense_layer_neuron_num = 128
# dropout_level = 0.1
# weight_decay = 0.0001
# class_num = 2

for experiment in experiments:
    f1_emwin = np.zeros([len(em_list),4])
    precision_emwin = np.zeros([len(em_list),4])
    recall_emwin = np.zeros([len(em_list),4])
    auc_std = np.zeros([len(em_list),4])
    auc_emwin = np.zeros([len(em_list),4])           #comment AUC metric lines for org. ER code
    # Loading data
    if (experiment == 'Gender_Clean_HI' or experiment == 'Emotion_Clean_HI'):
        matContent = sio.loadmat(dataPath + 'eeg_hi_cor.mat')
        features = matContent['hi_mat']
        labels = np.squeeze(matContent['gr_hi_label'])
        er_lab = np.squeeze(matContent['er_hi_label'])
    elif (experiment == 'Gender_notClean_HI' or experiment == 'Emotion_notClean_HI'):
        matContent = sio.loadmat(dataPath + 'eeg_hi_cor_notClean.mat')
        features = matContent['hi_mat']
        labels = np.squeeze(matContent['gr_hi_label'])
        er_lab = np.squeeze(matContent['er_hi_label'])        
    elif (experiment == 'Gender_Clean_LI' or experiment == 'Emotion_Clean_LI'):
        matContent = sio.loadmat(dataPath + 'eeg_li_cor.mat')
        features = matContent['li_mat']
        labels = np.squeeze(matContent['gr_li_label'])
        er_lab = np.squeeze(matContent['er_li_label'])
    elif (experiment == 'Gender_notClean_LI' or experiment == 'Emotion_notClean_LI'):
        matContent = sio.loadmat(dataPath + 'eeg_li_cor_notClean.mat')
        features = matContent['li_mat']
        labels = np.squeeze(matContent['gr_li_label'])
        er_lab = np.squeeze(matContent['er_li_label'])
    elif (experiment == 'Gender_Clean_EYE' or experiment == 'Emotion_Clean_EYE'):
        matContent = sio.loadmat(dataPath + 'eeg_eye_hi_cor.mat')
        features = matContent['eye_mat']
        labels = np.squeeze(matContent['gr_eye_label'])
        er_lab = np.squeeze(matContent['er_eye_label'])
    elif (experiment == 'Gender_notClean_EYE' or experiment == 'Emotion_notClean_EYE'):
        matContent = sio.loadmat(dataPath + 'eeg_eye_hi_cor_notClean.mat')
        features = matContent['eye_mat']
        labels = np.squeeze(matContent['gr_eye_label'])
        er_lab = np.squeeze(matContent['er_eye_label'])        
    elif (experiment == 'Gender_Clean_MOUTH' or experiment == 'Emotion_Clean_MOUTH'):
        matContent = sio.loadmat(dataPath + 'eeg_mouth_hi_cor.mat')
        features = matContent['mouth_mat']
        labels = np.squeeze(matContent['gr_mouth_label'])
        er_lab = np.squeeze(matContent['er_mouth_label'])
    elif (experiment == 'Gender_notClean_MOUTH' or experiment == 'Emotion_notClean_MOUTH'):
        matContent = sio.loadmat(dataPath + 'eeg_mouth_hi_cor_notClean.mat')
        features = matContent['mouth_mat']
        labels = np.squeeze(matContent['gr_mouth_label'])
        er_lab = np.squeeze(matContent['er_mouth_label'])
    # features = matContent['features']
    if em_wise:
        for w in range(0,4):
            for em in range(len(em_list)):
                print 'window-{0} emotion-{1}'.format(w,em_list[em])
                print "Training on Emotion-"+em_list[em]
                idx = np.where(er_lab == em+1)
                fold_features = np.squeeze(features[idx,])
                fold_features = fold_features[:,w*32:((w+1)*32),:]
                # print fold_features.shape,len(range(w*32,((w+1)*32)))
                fold_labels = np.squeeze(labels[idx,])
                rand_order = np.arange(fold_features.shape[0])
                np.random.shuffle(rand_order)
                fold_features = fold_features[rand_order,]
                fold_labels = np.squeeze(fold_labels[rand_order,])
                # print "features shape={0}".format(features.shape)
                # print "labels shape={0}".format(labels.shape)
                if (experiment[0][0] == 'G'):
                    # labels = np.squeeze(matContent['labels_gender'])
                    f1Net = np.zeros([foldNum,])
                    precisionNet = np.zeros([foldNum,])
                    recallNet = np.zeros([foldNum,])
                    accNet = np.zeros([foldNum,])
                    aucNet = np.zeros([foldNum,])           #comment AUC metric lines for org. ER code

                elif (experiment[0][0] == 'E'):
                    fold_labels = np.squeeze(matContent['labels_emotion']) - 1 
                    accNet = np.zeros([foldNum,])
                    aucNet = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
                # del matContent
                channels = features.shape[2]
                if (experiment[0][0] == 'E'):# valence label modification
                    # print np.where(labels==0)[0].tolist() + np.where(labels==1)[0].tolist()
                    fold_labels[np.where(fold_labels==0)[0].tolist() + np.where(fold_labels==1)[0].tolist() + np.where(fold_labels==2)[0].tolist() + np.where(fold_labels==4)[0].tolist(),] = 0
                    fold_labels[np.where(fold_labels == 3)[0].tolist() + np.where(fold_labels == 5)[0].tolist(),] = 1
                class_num = np.size(np.unique(fold_labels))
                labels_categorical = np_utils.to_categorical(fold_labels, class_num)    
                skf = StratifiedKFold(n_splits = foldNum)
                f = 0
                for train, test in skf.split(fold_features, fold_labels):
                    # Filling the missing values and Normalization
                    # print('Data Prepration ... \n')
                    trainingFeatures = fold_features[train,:,:]
                    testFeatures = fold_features[test,:,:]
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
                    # print 'training data shape-{0}'.format(trainingFeatures.shape)
                    testFeatures = scaler.transform(testFeatures)
                    testFeatures = np.reshape(testFeatures,[test_shape[0], test_shape[1], test_shape[2]])
                    # Training and evaluations
                    if (classifier == 'CNN' or classifier == 'Both'):
                        # CNN
                        # print('Training the CNN Model ... \n')
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
                        earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
                        model_save = ModelCheckpoint(experiment+'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
                        for ep in range(len(nb_epoch)):
                            sgd = SGD(lr=learn_rate / 10 ** ep, momentum = mtm, decay = weight_decay, nesterov = True)
                            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                            model.fit(trainingFeatures, labels_categorical[train,:], batch_size=batch_size, epochs=nb_epoch[ep],
                                      verbose=0, callbacks=[earlyStopping], validation_split=0.1) 
                        # sgd = adam()
                        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                        # model.summary()
                        # model.fit(trainingFeatures, labels_categorical[train,:], batch_size=batch_size, epochs=nb_epoch,
                        #           verbose=2, callbacks=[earlyStopping], validation_split=0.1) 
                        predicted_labelsNet = model.predict_classes(testFeatures, verbose=0)
                        predicted_probsNet = model.predict_proba(testFeatures,batch_size=1,verbose=0)
                        if (experiment[0][0] == 'G'):
                            precisionNet[f] = precision_score(fold_labels[test,], predicted_labelsNet, average='macro')
                            recallNet[f] = recall_score(fold_labels[test,], predicted_labelsNet, average='macro')
                            f1Net[f] = f1_score(fold_labels[test,], predicted_labelsNet, average='macro')
                            accNet[f] = accuracy_score(fold_labels[test,], predicted_labelsNet)
                            aucNet[f] = roc_auc_score(fold_labels[test,], predicted_probsNet[:,1])
                            print(experiment + '_CNN: Fold %d : f1_score: %.4f' % (f + 1, f1Net[f]))
                            print(experiment + '_CNN: Fold %d : auc: %.4f' % (f + 1, aucNet[f]))
                        elif (experiment[0][0] == 'E'):
                            accNet[f] = accuracy_score(fold_labels[test,], predicted_labelsNet)
                            aucNet[f] = roc_auc_score(fold_labels[test,], predicted_probsNet[:,1])
                            print(experiment + '_CNN: Fold %d : acc: %.4f' % (f + 1, accNet[f]))
                            print(experiment + '_CNN: Fold %d : auc: %.4f' % (f + 1, aucNet[f]))
                    f += 1
            # Saving the results
                del fold_features, fold_labels, model, scaler, imputer
                if (classifier == 'CNN'):
                    if (experiment[0][0] == 'G'):
                        sio.savemat(savePath + experiment + '_win'+str(w)+'_'+em_list[em] + '.mat', {'precisionNet': precisionNet,'recallNet': recallNet, 'f1Net': f1Net, 'accNet': accNet, 'aucNet': aucNet})

                    elif (experiment == 'Emotion_Clean_HI' or experiment == 'Emotion_notClean_HI'):
                        sio.savemat(savePath + experiment + classifier + '_Results_HI_'+ em_list[em] + '.mat', {'accNet': accNet, 'aucNet': aucNet})
                auc_emwin[em,w] = np.mean(aucNet)
                auc_std[em,w] = np.std(aucNet)
        del matContent,features,aucNet,precisionNet,recallNet,f1Net,accNet
    sio.savemat(savePath + experiment + '_win.mat', {'auc_emwin': auc_emwin,'auc_std':auc_std})
