# Genderwise valence recognition with gaze features and AdaBoost for EYE/MOUTH mask conditions
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

dataPath = '/home/maneesh/atom/masked_data/'
savePath = '/home/maneesh/Downloads/DNN/Results/'
# female=0, male=1 in /home/maneesh/atom/Plos_code/feature_vector.mat
experiments = ['Emotion_female_EYE','Emotion_female_MOUTH','Emotion_male_EYE','Emotion_male_MOUTH'] # 
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


gaze_VR_results2_DL={experiments[0]:None,experiments[1]:None,experiments[2]:None,experiments[3]:None,}
for experiment in experiments:
	f1Net[experiment] = np.zeros([foldNum,])
	precisionNet[experiment] = np.zeros([foldNum,])
	recallNet[experiment] = np.zeros([foldNum,])
	accNet[experiment] = np.zeros([foldNum,])
	aucNet[experiment] = np.zeros([foldNum,])#comment AUC metric lines for org. ER code
	accSVM[experiment] = np.zeros([foldNum,])
	aucSVM[experiment] = np.zeros([foldNum,])#comment AUC metric lines for org. ER code

	if experiment==experiments[0]:
		matContent = sio.loadmat(dataPath+'female_feat')
		# print matContent.keys()
		features = matContent['featureMat']
		labels = matContent['labelMat']
		idx = np.squeeze(np.intersect1d(np.where(labels[:,4]==1),np.squeeze(np.intersect1d(np.where(labels[:,0]==1),np.where(labels[:,1]==labels[:,3])))))
		features = features[idx,]
		labels = labels[idx,3]-1
		labels[np.where(labels==0)[0].tolist() + np.where(labels==1)[0].tolist() + np.where(labels==2)[0].tolist() + np.where(labels==4)[0].tolist(),] = 0
		labels[np.where(labels == 3)[0].tolist() + np.where(labels == 5)[0].tolist(),] = 1
	elif experiment==experiments[1]:
		matContent = sio.loadmat(dataPath+'female_feat')
		features = matContent['featureMat']
		labels = matContent['labelMat']
		idx = np.squeeze(np.intersect1d(np.where(labels[:,4]==1),np.squeeze(np.intersect1d(np.where(labels[:,0]==1),np.where(labels[:,1]==labels[:,3])))))
		features = features[idx,]
		labels = labels[idx,3]-1
		labels[np.where(labels==0)[0].tolist() + np.where(labels==1)[0].tolist() + np.where(labels==2)[0].tolist() + np.where(labels==4)[0].tolist(),] = 0
		labels[np.where(labels == 3)[0].tolist() + np.where(labels == 5)[0].tolist(),] = 1
	elif experiment==experiments[2]:
		matContent = sio.loadmat(dataPath+'male_feat')
		features = matContent['featureMat']
		labels = matContent['labelMat']
		idx = np.squeeze(np.intersect1d(np.where(labels[:,4]==1),np.squeeze(np.intersect1d(np.where(labels[:,0]==1),np.where(labels[:,1]==labels[:,3])))))
		features = features[idx,]
		labels = labels[idx,3]-1
		labels[np.where(labels==0)[0].tolist() + np.where(labels==1)[0].tolist() + np.where(labels==2)[0].tolist() + np.where(labels==4)[0].tolist(),] = 0
		labels[np.where(labels == 3)[0].tolist() + np.where(labels == 5)[0].tolist(),] = 1
	elif experiment==experiments[3]:
		matContent = sio.loadmat(dataPath+'male_feat')
		features = matContent['featureMat']
		labels = matContent['labelMat']
		idx = np.squeeze(np.intersect1d(np.where(labels[:,4]==0),np.squeeze(np.intersect1d(np.where(labels[:,0]==1),np.where(labels[:,1]==labels[:,3])))))
		features = features[idx,]
		labels = labels[idx,3]-1
		labels[np.where(labels==0)[0].tolist() + np.where(labels==1)[0].tolist() + np.where(labels==2)[0].tolist() + np.where(labels==4)[0].tolist(),] = 0
		labels[np.where(labels == 3)[0].tolist() + np.where(labels == 5)[0].tolist(),] = 1

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
		earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
		for ep in range(0,len(nb_epoch)):
			sgd = SGD(lr=learn_rate / 10 ** ep, momentum = mtm, decay = weight_decay, nesterov = True)
			model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
			model.fit(trainfeatures, np_utils.to_categorical(trainlabels, class_num), batch_size=batch_size, epochs=nb_epoch[ep],
			verbose=2,  validation_split=0.1, callbacks=[earlyStopping])
		# feat_imp = np.concatenate((feat_imp,np.reshape(model.feature_importances_,[1,825])),axis=0)
		# model.predict_proba(testfeatures,batch_size=1,verbose=0)
		predicted_labelsNet = model.predict_classes(testfeatures)
		predicted_probsNet = model.predict_proba(testfeatures)
		accNet[experiment][f] = accuracy_score(testlabels, predicted_labelsNet)
		aucNet[experiment][f] = roc_auc_score(testlabels, predicted_probsNet[:,1])
		f1Net[experiment][f] = f1_score(testlabels, predicted_labelsNet, average='macro')
		precisionNet[experiment][f] = precision_score(testlabels, predicted_labelsNet, average='macro')
		recallNet[experiment][f] = recall_score(testlabels, predicted_labelsNet, average='macro')
		print(experiment + '_CNN: Fold %d : acc: %.4f' % (f + 1, accNet[experiment][f]))
		print(experiment + '_CNN: Fold %d : auc: %.4f' % (f + 1, aucNet[experiment][f]))
		print(experiment + '_CNN: Fold %d : f1: %.4f' % (f + 1, f1Net[experiment][f]))
		f+=1
	del model,features, labels

	print '\n'+experiment+'-{0}+-{1} \n'.format(np.mean(aucNet[experiment]),np.std(aucNet[experiment]))
	gaze_VR_results2_DL[experiment] = {'auc':aucNet[experiment],'acc':accNet[experiment],'f1':f1Net[experiment]}
sio.savemat('gaze_VR_results2_DL',gaze_VR_results2_DL)
