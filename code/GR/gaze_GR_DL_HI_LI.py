# Emotionwise gender recognition with gaze features and AdaBoost for EYE/MOUTH mask conditions
import numpy as np
import scipy.io as sio
import csv
from sklearn.ensemble import AdaBoostClassifier
from keras.layers import Conv1D, AveragePooling1D,MaxPooling1D, Input
from keras.layers.normalization import BatchNormalization
from keras.callbacks  import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,roc_auc_score, roc_curve, auc
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD,adam, Nadam, Adamax, Adadelta, RMSprop, Adagrad
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from keras.utils.layer_utils import print_summary

dataPath = './data/feature_vector'
savePath = './Results/'
# female=0, male=1 in /home/maneesh/atom/Plos_code/feature_vector.mat
experiments = ['Gender_HI','Gender_LI']#,'Gender_EYE','Gender_MOUTH'] # 
emotion = ['Anger','Disgust','Fear','Happy','Sad','Surprise']
foldNum = 10
gen_wise = 1
test_quant = 0.1

skf = StratifiedKFold(n_splits = foldNum)
f1Net = {}
precisionNet = {}
recallNet = {}
accNet = {}
aucNet = {}
accSVM = {}
aucSVM = {}

nb_filters = [16, 32, 32, 64]
kernel_size = 3
pool_size = 2
stride_size = 2
nb_epoch = [20, 20, 20]  
learn_rate = 0.00176
batch_size = 16
mtm = 0.9
padding = 'same'
dense_layer_neuron_num = 128
dropout_level = 0.1
weight_decay = 0.0001
class_num = 2
input_feat_size = (825,1)

fieldnames = ['Experiment', 'Emotion','Estimators','aucNet','f1Net','accNet','precisionNet', 'recallNet']

def model_def(nb_filters,kernel_size,padding,input_shape,trainable,dropout_level,class_num):
    model = Sequential()
    model.add(Conv1D(filters=nb_filters[0], kernel_size=kernel_size, padding=padding, activation='relu',
                 input_shape=input_shape, trainable=trainable))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dropout(dropout_level))
    model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    model.add(Conv1D(filters=nb_filters[1], kernel_size=kernel_size, padding=padding, activation='relu',
                 kernel_initializer='he_normal', trainable=trainable))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dropout(dropout_level))
    model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    model.add(Conv1D(filters=nb_filters[2], kernel_size=kernel_size, padding=padding, activation='relu',
                 kernel_initializer='he_normal', trainable=trainable))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dropout(dropout_level))
    model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))

    model.add(Flatten())
    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dense(dense_layer_neuron_num, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(dropout_level))
    model.add(Dense(class_num))
    model.add(Activation('softmax'))

    return model

with open('GR_gaze_results1_DL_log.csv', 'a') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()

for n_est in range(10,100,10):
	for em in range(1,7):
		for experiment in experiments:
			f1Net[experiment] = np.zeros([foldNum,])
			precisionNet[experiment] = np.zeros([foldNum,])
			recallNet[experiment] = np.zeros([foldNum,])
			accNet[experiment] = np.zeros([foldNum,])
			aucNet[experiment] = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
			accSVM[experiment] = np.zeros([foldNum,])
			aucSVM[experiment] = np.zeros([foldNum,])#comment AUC metric lines for org. ER code

			matContent = sio.loadmat(dataPath)
			# print matContent.keys()
			# print allgenderMat.shape,allfeatures.shape,alllabels.shape
			if experiment==experiments[0]:
				allfeatures = matContent['featureMat']
				alllabels = matContent['labelMat']
				allgenderMat = np.squeeze(matContent['genderMat'])
				idx = np.squeeze(np.intersect1d(np.where(alllabels[:,3]==em),np.squeeze(np.intersect1d(np.where(alllabels[:,0]==1),np.where(alllabels[:,1]==alllabels[:,3])))))
				features = allfeatures[idx,]
				labels = allgenderMat[idx]
			elif experiment==experiments[1]:
				allfeatures = matContent['featureMat']
				alllabels = matContent['labelMat']
				allgenderMat = np.squeeze(matContent['genderMat'])
				idx = np.squeeze(np.intersect1d(np.where(alllabels[:,3]==em),np.squeeze(np.intersect1d(np.where(alllabels[:,0]==1),np.where(alllabels[:,1]==alllabels[:,3])))))
				features = allfeatures[idx,]
				labels = allgenderMat[idx]
			# elif experiment==experiments[2]:
			# 	allfeatures = matContent['featureMat']
			# 	alllabels = matContent['labelMat']
			# 	allgenderMat = np.squeeze(matContent['genderMat'])
			# 	features = allfeatures[np.squeeze(np.intersect1d(np.where(alllabels[:,0]==1),np.where(alllabels[:,1]==alllabels[:,3]))),]
			# 	labels = allgenderMat[np.squeeze(np.intersect1d(np.where(alllabels[:,0]==1),np.where(alllabels[:,1]==alllabels[:,3]))),3]-1
			# elif experiment==experiments[3]:
			# 	features = features[np.squeeze(np.intersect1d(np.where(alllabels[:,0]==0),np.where(alllabels[:,1]==alllabels[:,3]))),]
			# 	labels = allgenderMat[np.squeeze(np.intersect1d(np.where(alllabels[:,0]==0),np.where(alllabels[:,1]==alllabels[:,3]))),3]-1

			rand_order = np.arange(features.shape[0])
			np.random.shuffle(rand_order)
			features = features[rand_order,]
			labels = np.squeeze(labels[rand_order,])
			class_num = np.size(np.unique(labels))
			# print class_num
			labels_categorical = np_utils.to_categorical(labels, class_num)    
			imputer = Imputer()
			scalar = StandardScaler()
			# print features.shape, labels.shape
			f=0
			for train, test in skf.split(features, labels):
				trainfeatures = features[train,]
				trainlabels = labels[train,]
				testfeatures = features[test,]
				testlabels = labels[test,]
				imputer.fit(trainfeatures)
				trainfeatures = imputer.transform(trainfeatures)
				scalar.fit(trainfeatures)
				trainfeatures = scalar.transform(trainfeatures)
				testfeatures = imputer.transform(testfeatures)
				testfeatures = scalar.transform(testfeatures)
				trainfeatures = np.reshape(trainfeatures,[trainfeatures.shape[0],trainfeatures.shape[1],1])
				testfeatures = np.reshape(testfeatures,[testfeatures.shape[0],testfeatures.shape[1],1])
				model = model_def(nb_filters=nb_filters, kernel_size=kernel_size, padding=padding, input_shape=(825,1), trainable=True, dropout_level=dropout_level, class_num=class_num)
				earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
				for ep in range(0,len(nb_epoch)):
					sgd = SGD(lr=learn_rate / 10 ** ep, momentum = mtm, decay = weight_decay, nesterov = True)
					model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
					model.fit(trainfeatures, np_utils.to_categorical(trainlabels, class_num), batch_size=batch_size, epochs=nb_epoch[ep],
					verbose=0,  validation_split=0.1, callbacks=[earlyStopping])
				# feat_imp = np.concatenate((feat_imp,np.reshape(model.feature_importances_,[1,825])),axis=0)
				# model.predict_proba(testfeatures,batch_size=1,verbose=0)
				predicted_labelsNet = model.predict_classes(testfeatures)
				predicted_probsNet = model.predict_proba(testfeatures,batch_size=1,verbose=0)
				accNet[experiment][f] = accuracy_score(testlabels, predicted_labelsNet)
				aucNet[experiment][f] = roc_auc_score(testlabels, predicted_probsNet[:,1])
				f1Net[experiment][f] = f1_score(testlabels, predicted_labelsNet, average='macro')
				precisionNet[experiment][f] = precision_score(testlabels, predicted_labelsNet, average='macro')
				recallNet[experiment][f] = recall_score(testlabels, predicted_labelsNet, average='macro')
				# print(experiment +'\t'+emotion[em-1]+ '_CNN: Fold %d : acc: %.4f' % (f + 1, accNet[experiment][f]))
				# print(experiment +'\t'+emotion[em-1]+ '_CNN: Fold %d : auc: %.4f' % (f + 1, aucNet[experiment][f]))
				# print(experiment +'\t'+emotion[em-1]+ '_CNN: Fold %d : f1: %.4f' % (f + 1, f1Net[experiment][f]))
				f+=1
			del model,features, labels

			print '\n'+experiment+'\t'+emotion[em-1]+'-{0}+-{1} \n'.format(np.mean(aucNet[experiment]),np.std(aucNet[experiment]))
			with open('GR_gaze_results1_DL_log.csv', 'a') as csvfile:
				writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
				writer.writerow({'Experiment':experiment, 'Emotion':emotion[em-1],'Estimators':n_est,'aucNet': str(np.mean(aucNet[experiment]))+'+-'+str(np.std(aucNet[experiment])),
				'f1Net': str(np.mean(f1Net[experiment]))+'+-'+str(np.std(f1Net[experiment])),'accNet': str(np.mean(accNet[experiment]))+'+-'+str(np.std(accNet[experiment]))
				,'precisionNet': str(np.mean(precisionNet[experiment]))+'+-'+str(np.std(precisionNet[experiment])),'recallNet': str(np.mean(recallNet[experiment]))+'+-'+str(np.std(recallNet[experiment]))})

