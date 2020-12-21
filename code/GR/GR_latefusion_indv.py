#late fusion results for mask type wise-HI,LI,EYE,MOUTH
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
from sklearn.externals import joblib

dataPath = '/home/maneesh/journal_fusion/gaze_based_eeg_selected/'
savePath = '/home/maneesh/Downloads/DNN/Results/'
experiments = ['Gender_notClean_HI', 'Gender_notClean_LI', 'Gender_notClean_EYE', 'Gender_notClean_MOUTH'] #  
# experiments = [ 'Gender_notClean_EYE', 'Gender_notClean_MOUTH'] #  ]
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
f1train = {}
skf = StratifiedKFold(foldNum)
alpha_range = np.linspace(0.01,1,num=100)

# for experiment in experiments:
#     f1Net[experiment] = np.zeros([foldNum,])
#     precisionNet[experiment] = np.zeros([foldNum,])
#     recallNet[experiment] = np.zeros([foldNum,])
#     accNet[experiment] = np.zeros([foldNum,])
#     aucNet[experiment] = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
#     accAda[experiment] = np.zeros([foldNum,])
#     aucAda[experiment] = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
#     f1Ada[experiment] = np.zeros([foldNum,])
#     precisionAda[experiment] = np.zeros([foldNum,])
#     recallAda[experiment] = np.zeros([foldNum,])
#     aucBoth[experiment] = np.zeros([foldNum,])

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

def late_fus(pe,pg,alpha,f1_e,f1_g):
    te_prime = (f1_e)/((alpha*f1_e)+((1-alpha)*f1_g))
    tg_prime = (f1_g)/((alpha*f1_e)+((1-alpha)*f1_g))
    te,tg = te_prime/(te_prime+tg_prime),tg_prime/(te_prime+tg_prime)
    return te,tg#te=1,tg=1 is equal weight fusion
    # return 1,1

if __name__ == '__main__':
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
        f1train = 0
        features_eeg,features_gaze,labels,em_label = data_load_casebasis(experiment)
        rand_order = np.arange(features_eeg.shape[0])
        np.random.shuffle(rand_order)
        features_eeg = features_eeg[rand_order,]
        features_gaze = features_gaze[rand_order,]
        labels = np.squeeze(labels[rand_order,])
        class_num = np.size(np.unique(labels))
        # idx = np.squeeze(np.where(em_label_all==em))
        # # print len(idx)
        # features_eeg = features_eeg_all[idx,:,:]
        # features_gaze = features_gaze_all[idx,:]
        # labels = np.squeeze(labels_all[idx,])
        f = 0
        # print features_eeg.shape,labels.shape,idx.shape
        for train,test in skf.split(features_eeg,labels):
            alpha = alpha_range[0]
            f1_best = 0
            # for alpha in alpha_range:
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
            # split_name = experiment.split('_')
            dl_model.save_weights('./GR_fusion_indv_models/'+experiment+'fold-'+str(f)+'_weights.hdf5')#toggle between Clean and notClean here
            earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
        # model_save = ModelCheckpoint(experiment+'weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            for ep in range(len(nb_epoch)):
                sgd = SGD(lr=learn_rate / 10 ** ep, momentum = mtm, decay = weight_decay, nesterov = True)
                dl_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                dl_model.fit(train_features, np_utils.to_categorical(labels[train,], class_num), batch_size=batch_size, epochs=nb_epoch[ep],
                      verbose=0, callbacks=[earlyStopping], validation_split=0.1) 
            # # for ep in nb_epoch:
            #     sgd = SGD(lr=learn_rate / 10 ** ep, momentum = mtm, decay = weight_decay, nesterov = True)
            #     dl_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            #     dl_model.fit(train_features, np_utils.to_categorical(labels[train,], class_num), batch_size=batch_size, epochs=ep,
            #         verbose=2,  validation_split=0.1, callbacks=[earlyStopping])
            predicted_trainprobsNet = dl_model.predict_proba(train_features, batch_size=1, verbose=0)
            predicted_trainlabelsNet = dl_model.predict_classes(train_features, verbose=0)
            f1_Net = recall_score(labels[train,], predicted_trainlabelsNet, average='macro')
            
            predicted_labelsNet = dl_model.predict_classes(test_features, verbose=0)
            predicted_probsNet = dl_model.predict_proba(test_features,batch_size=1,verbose=0)
            accNet[experiment][f] = accuracy_score(labels[test,], predicted_labelsNet)
            # print np.unique(predicted_probsNet)
            aucNet[experiment][f] = roc_auc_score(labels[test,], predicted_probsNet[:,1])
            f1Net[experiment][f] = f1_score(labels[test,], predicted_labelsNet, average='macro')
            precisionNet[experiment][f] = precision_score(labels[test,], predicted_labelsNet, average='macro')
            recallNet[experiment][f] = recall_score(labels[test,], predicted_labelsNet, average='macro')
            # print(experiment + '_CNN: Fold %d : acc: %.4f' % (f + 1, accNet[experiment][f]))
            # print(experiment + '_CNN: Fold %d : auc: %.4f' % (f + 1, aucNet[experiment][f]))
            # print(experiment + '_CNN: Fold %d : f1: %.4f' % (f + 1, f1Net[experiment][f]))
            train_features = features_gaze[train,]
            test_features = features_gaze[test,]
            # print train_features.shape
            imputer.fit(train_features)
            train_features = imputer.transform(train_features)
            test_features = imputer.transform(test_features)
            scalar = StandardScaler()
            scalar.fit(train_features)
            train_features = scalar.transform(train_features)
            test_features = scalar.transform(test_features)
            ada_model = AdaBoostClassifier(algorithm='SAMME',n_estimators = n_est, random_state=245641)
            ada_model.fit(train_features,labels[train,])
            joblib.dump(ada_model,'./GR_fusion_indv_models/'+experiment+'fold-'+str(f)+'_model.pkl')
            predicted_trainprobsAda = ada_model.predict_proba(train_features)
            predicted_trainlabelsAda = ada_model.predict(train_features)
            f1_Ada = recall_score(labels[train,], predicted_trainlabelsAda, average='macro')
            # x = input('enter')
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

                #######Get fused f1 score on training set for best alpha
            for alpha in alpha_range:
                te,tg = late_fus(predicted_trainprobsNet, predicted_trainprobsAda, alpha, f1_Net, f1_Ada)
                po_train_unnorm = (alpha*(predicted_trainprobsNet*te))+((1-alpha)*predicted_trainprobsAda*tg)
                po_train = np.empty(po_train_unnorm.shape)#po_train_unnorm#
                po_train[:,0] = np.divide(po_train_unnorm[:,0],po_train_unnorm[:,0]+po_train_unnorm[:,1])
                po_train[:,1] = np.divide(po_train_unnorm[:,1],po_train_unnorm[:,0]+po_train_unnorm[:,1])
                f1train = f1_score(labels[train,], po_train[:,1]>po_train[:,0],average='macro')
                if f1train>f1_best:
                    alpha_best = alpha
                    f1_best = f1train
                # print len(test)
            te,tg = late_fus(predicted_probsNet, predicted_probsAda, alpha_best, f1_Net, f1_Ada)
            po_unnorm = alpha_best*(predicted_probsNet*te)+(1-alpha_best)*predicted_probsAda*tg
            po_test = np.empty(po_unnorm.shape)#po_unnorm#
            po_test[:,0] = np.divide(po_unnorm[:,0],po_unnorm[:,0]+po_unnorm[:,1])
            po_test[:,1] = np.divide(po_unnorm[:,1],po_unnorm[:,0]+po_unnorm[:,1])
            # print po#exceeding 1
            # x = input('enter')
            accBoth[experiment][f] = accuracy_score(labels[test,], po_test[:,1]>po_test[:,0])
            aucBoth[experiment][f] = roc_auc_score(labels[test,], po_test[:,1])
            f1Both[experiment][f] = f1_score(labels[test,], po_test[:,1]>po_test[:,0],average='macro')
            print 'Best alpha-%.2f auc-%.4f'%(alpha_best,aucBoth[experiment][f])
            # accBoth[experiment][f] = accuracy_score(labels[test,], (predicted_probsAda[:,1]+predicted_probsNet[:,1])/2>0.5)
            # aucBoth[experiment][f] = roc_auc_score(labels[test,], (predicted_probsAda[:,1]+predicted_probsNet[:,1])/2)
            # f1Both[experiment][f] = f1_score(labels[test,], (predicted_probsAda[:,1]+predicted_probsNet[:,1])/2>0.5,average='macro')
            f+=1
        # print(experiment + '_CNN: Mean %.4f : acc: %.4f' % (np.mean(accNet[experiment]),np.std(accNet[experiment])))
        print(experiment + '_CNN: Mean %.4f : auc: %.4f' % (np.mean(aucNet[experiment]),np.std(aucNet[experiment])))
        # print(experiment + '_CNN: Mean %.4f : f1: %.4f' % (np.mean(f1Net[experiment]),np.std(f1Net[experiment])))

        # print(experiment + '_Ada: Mean %.4f : acc: %.4f' % (np.mean(accAda[experiment]),np.std(accAda[experiment])))
        print(experiment + '_Ada: Mean %.4f : auc: %.4f' % (np.mean(aucAda[experiment]),np.std(aucAda[experiment])))
        # print(experiment + '_Ada: Mean %.4f : f1: %.4f' % (np.mean(f1Ada[experiment]),np.std(f1Ada[experiment])))
        print(experiment+ '_Both: Mean %.4f : auc: %.4f' % (np.mean(aucBoth[experiment]),np.std(aucBoth[experiment])))
        # sio.savemat(experiment+emotion[em]+'_fuse.mat', {'accNet': accNet[experiment], 'accAda': accAda[experiment], 'accBoth': accBoth[experiment], 'aucNet': accNet[experiment], 'aucAda': aucAda[experiment], 'aucBoth': aucBoth[experiment]
        #     , 'f1Net': f1Net[experiment], 'f1Ada': f1Ada[experiment], 'f1Both': f1Both[experiment]})
        sio.savemat('./GR_fusion_indv_results/'+experiment+'_forfusion_indv.mat', {'accNet': accNet[experiment], 'accAda': accAda[experiment], 'accBoth': accBoth[experiment], 'aucNet': accNet[experiment], 'aucAda': aucAda[experiment], 'aucBoth': aucBoth[experiment]
            , 'f1Net': f1Net[experiment], 'f1Ada': f1Ada[experiment], 'f1Both': f1Both[experiment],'labels':labels[test,],'prob_Net':predicted_probsNet,'prob_Ada':predicted_probsAda})