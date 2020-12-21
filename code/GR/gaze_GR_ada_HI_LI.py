# Emotionwise gender recognition with gaze features and AdaBoost for HI/LI mask conditions
import numpy as np
import scipy.io as sio
import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from keras.utils import np_utils

dataPath = './data/feature_vector'
savePath = './Results/'
# female=0, male=1 in /home/maneesh/atom/Plos_code/feature_vector.mat
experiments = ['Gender_HI','Gender_LI']
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

fieldnames = ['Experiment', 'Emotion','Estimators','aucNet','f1Net','accNet','precisionNet', 'recallNet']

with open('GR_gaze_results1_log.csv', 'a') as csvfile:
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
				model = AdaBoostClassifier(algorithm='SAMME',n_estimators = n_est, random_state=245641)
				model.fit(trainfeatures,trainlabels)
				# print model.estimators_
				# model.predict_proba(testfeatures)
				predicted_labelsNet = model.predict(testfeatures)
				predicted_probsNet = model.predict_proba(testfeatures)
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
			with open('GR_gaze_results1_log.csv', 'a') as csvfile:
				writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
				writer.writerow({'Experiment':experiment, 'Emotion':emotion[em-1],'Estimators':n_est,'aucNet': str(np.mean(aucNet[experiment]))+'+-'+str(np.std(aucNet[experiment])),
				'f1Net': str(np.mean(f1Net[experiment]))+'+-'+str(np.std(f1Net[experiment])),'accNet': str(np.mean(accNet[experiment]))+'+-'+str(np.std(accNet[experiment]))
				,'precisionNet': str(np.mean(precisionNet[experiment]))+'+-'+str(np.std(precisionNet[experiment])),'recallNet': str(np.mean(recallNet[experiment]))+'+-'+str(np.std(recallNet[experiment]))})

