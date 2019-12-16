import os
import cv2
import pickle
import numpy as np
import pdb
import requests
from collections import defaultdict
import random 
import time
import argparse
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tqdm import *

from functools import wraps
from time import time as _timenow 
from sys import stderr


#### Loading CIFAR 10 DATA 
def load_cifar():

    trn_data, trn_labels, tst_data, tst_labels = [], [], [], []
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data
    
    for i in trange(5):
        batchName = './data/data_batch_{0}'.format(i + 1)
        unpickled = unpickle(batchName)
        trn_data.extend(unpickled['data'])
        trn_labels.extend(unpickled['labels'])
    unpickled = unpickle('./data/test_batch')
    tst_data.extend(unpickled['data'])
    tst_labels.extend(unpickled['labels'])
    return trn_data, trn_labels, tst_data, tst_labels

#### Preprocessing the Images
def image_prep(image):
	m = np.mean(image,axis = 0)
	sd = np.std(image,axis = 0)
	preprocessed_image = (image - m)/sd
	return preprocessed_imagez

#### Reducing dimension using LDA or PCA
def reduce_dim(type_of_reduction,train,val,test,param,train_label):
	if (type_of_reduction == 'RAW'):
		pass
	elif (type_of_reduction == 'PCA'):
		pca = PCA(param)
		pca.fit(train)
		train = pca.transform(train)
		val =  pca.transform(val)
		test = pca.transform(test)
	elif (type_of_reduction == 'LDA'):
		lda = LDA(n_components = param)
		train = lda.fit_transform(train,train_label)
		val = lda.transform(val)
		test = lda.transform(test)
	return train,val,test

#### Returns the trained classifier given training data
def classify(type_of_classifier,X,y,*argv):
	if (type_of_classifier == 'LR'):
		classifier = LogisticRegression(C = argv[0],solver = 'lbfgs',max_iter =argv[1],multi_class = 'multinomial',tol = argv[2])
		classifier.fit(X,y)
	if (type_of_classifier == 'LinearSVM'):
		classifier = LinearSVC(C = argv[0],max_iter = argv[1],tol = argv[2])
		classifier.fit(X,y)
	if (type_of_classifier == 'KernelSVM'):
		classifier = SVC(C = argv[0],kernel = 'rbf',degree = argv[1],gamma = 'auto',tol = argv[2])
		classifier.fit(X,y)
	if (type_of_classifier == 'MLP'):
		classifier = MLPClassifier(activation = argv[0],learning_rate_init = argv[1],alpha = argv[2],momentum = 0.9,beta_1 = 0.9,beta_2 = 0.99,early_stopping = argv[3],shuffle = True)
		classifier.fit(X,y)
	return classifier

#### Returns the f1 score and accuracy of the classifier
def evaluate(target,predicted):
	f1 = f1_score(target,predicted,average = 'micro')
	acc = accuracy_score(target,predicted)
	return f1,acc

#### Runs the given classifier on test data.
def test(clf,X,y):
	predictions = clf.predict(X)
	f1 , acc = evaluate(y,predictions)
	return f1,acc

#### Main function
def main():
	error_reduction = "Specify the kind of data reduction technique u want:\n RAW, PCA, LDA"
	error_classifier = "Specify the kind of classifier for classification: \n LinearSVM, KernelSVM, MLP, LR"
	parser = argparse.ArgumentParser()
	parser.add_argument("type_of_reduction",help=error_reduction,type=str)
	parser.add_argument("type_of_classifier",help=error_classifier,type=str)
	parser.add_argument("--C",help="Optional Argument: C",type=float, default = 1)
	parser.add_argument("--max_iter",help="Optional Argument: Max Iterations",type=int,default = 200)
	parser.add_argument("--tol",help="Optional Argument: Tolerance",type=float,default = 1e-3)
	parser.add_argument("--deg",help="Optional Argument: Degree of Kernel SVM",type=int,default = 3)
	parser.add_argument("--activation",help="Optional Argument: Activation Function",type=str,default = 'relu')
	parser.add_argument("--learning_rate",help="Optional Argument: Learning Rate",type=float,default = 0.001)
	parser.add_argument("--alpha",help="Optional Argument: L2 Penalty parameter",type=float,default = 0.0001)
	parser.add_argument("--early_stopping",help="Optional Argument: Early Stopping",type=bool,default = True)


	args = parser.parse_args()
	if (args.type_of_reduction != 'PCA' and args.type_of_reduction != 'LDA' and args.type_of_reduction != 'RAW'):
		print (error_reduction)
		return
	if (args.type_of_classifier != 'LinearSVM' and args.type_of_classifier != 'KernelSVM' and args.type_of_classifier != 'MLP' and args.type_of_classifier != 'LR'):
		print (error_classifier)
		return

	print('Running Algorithm')
	param = 1					
	trn_data, trn_labels, tst_data, tst_labels = load_cifar()
	trn_data = np.array(trn_data)
	tst_data = np.array(tst_data)
	print("Preprocessing Images")
	trn_data = image_prep(trn_data)
	tst_data = image_prep(tst_data)
	X_train, X_val, y_train, y_val = train_test_split(trn_data, trn_labels,test_size = 0.20) 

	print('Training Data Format: {}'.format(X_train.shape))
	if (args.type_of_reduction == 'PCA'):
		param = 0.95	### To select components such that variance is min 95%
	elif (args.type_of_reduction == 'LDA'):
		param = 9		### To select the number of components for LDA
	X_train,X_val,tst_data = reduce_dim(args.type_of_reduction,X_train,X_val,tst_data,param,y_train)
	print("Dimesionallity Reduction Completed")
	print('Training Data Format after dimesionallity reduction {}'.format(X_train.shape))

	print("Training the classifier")
	if (args.type_of_classifier == 'LR' or args.type_of_classifier == 'LinearSVM'):
		clf = classify(args.type_of_classifier,X_train,y_train,args.C,args.max_iter,args.tol)
	elif (args.type_of_classifier == 'KernelSVM'):
		clf = classify(args.type_of_classifier,X_train,y_train,args.C,args.deg,args.tol)
	elif (args.type_of_classifier == 'MLP'):
		clf = classify(args.type_of_classifier,X_train,y_train,args.activation,args.learning_rate,args.alpha,args.early_stopping)		
	print("Classifier trained")

	print("Testing the test data on trained classifier with given arguments")
	f_score , accuracy_ = test(clf,tst_data,tst_labels)
	print('Val - F1 score: {}\n Accuracy: {}'.format(f_score, accuracy_))

	
	if (args.type_of_reduction == 'RAW'):
		print("Hyperparameter tunning is not applied on RAW data because of large time taken.")
		return
	print("Hyperparameter Tuning using Grid Search on reduced data...")
	if (args.type_of_classifier == 'LR' or args.type_of_classifier == 'LinearSVM'):
		C_values = [0.0001,0.0005,0.001,0.01,0.05,0.1,0.5,1]
		scors = [0] * len(C_values)
		count = 0
		mx_score = 0
		mx_score_C = -1
		f = open('Results.txt','w')
		for i in C_values:
			clf = classify(args.type_of_classifier,X_train,y_train,i,args.max_iter,args.tol)
			f_score , accuracy_ = test(clf,tst_data,tst_labels)
			scors[count] =  f_score
			print('Iteration number :{}'.format(count))
			f.write('Iteration Number: {}  C value: {}  Accuracy : {}\n'.format(count,i,f_score))
			if (f_score >= mx_score):
				mx_score = f_score
				mx_score_C = C_values[count]
			count = count + 1
		f.close()
		print('Value of the Hyperparameter C is :{}'.format(mx_score_C))
		print('F1 score and Accuracy after Hyperparameter tunning are: ')
		clf_final = classify(args.type_of_classifier,X_train,y_train,mx_score_C,args.max_iter,args.tol)
		f_score , accuracy_ = test(clf_final,tst_data,tst_labels)
		print('Val - F1 score: {}\n Accuracy: {}'.format(f_score, accuracy_))
		plt.plot (C_values,scors)
		plt.xlabel('Hyperparameter C values')
		plt.ylabel('Accuracy')
		plt.title('Variation of Accuracy of {} classifier with Hyperparameter C on {} reduced data'.format(args.type_of_classifier,args.type_of_reduction))
		plt.show()

	elif(args.type_of_classifier == 'MLP'):
		learning_rate_values = [0.0001,0.001,0.01,0.05,0.1,0.5,0.9]
		alpha_values = [0.0001,0.001,0.01,0.1,0.5,1,5]
		t = (len(learning_rate_values) * len(alpha_values))
		scors = np.zeros((t,3))
		count = 0
		mx_score = 0
		mx_score_learning_rate = -1
		mx_score_alpha = -1
		f = open('Result.txt','w')
		for i in learning_rate_values:
			for j in alpha_values:
				clf = classify(args.type_of_classifier,X_train,y_train,args.activation,i,j,args.early_stopping)		
				f_score , accuracy_ = test(clf,tst_data,tst_labels)
				scors[count][0] = i
				scors[count][1] = j
				scors[count][2] = f_score
				f.write('Iteration: {}  Learning Rate: {}  Alpha: {}  Accuracy: {}\n'.format(count,i,j,f_score))
				print('Iteration number :{}'.format(count))
				if (f_score > mx_score):
					mx_score = f_score
					mx_score_learning_rate = i
					mx_score_alpha = j
				count = count + 1
		f.close()
		print('Value of the Hyperparameter learning_rate is :{}'.format(mx_score_learning_rate))
		print('Value of the Hyperparameter alpha is :{}'.format(mx_score_alpha))
		print('F1 score and Accuracy after Hyperparameter tunning are: ')
		clf_final = classify(args.type_of_classifier,X_train,y_train,args.activation,mx_score_learning_rate,mx_score_alpha,args.early_stopping)		
		f_score , accuracy_ = test(clf_final,tst_data,tst_labels)
		print('Val - F1 score: {}\n Accuracy: {}'.format(f_score, accuracy_))

	return 


if _name_ == 	'_main_':
	main()
