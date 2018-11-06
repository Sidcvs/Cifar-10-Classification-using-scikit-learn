import os
import cv2
import pickle
import numpy as np
import pdb
import requests
from collections import defaultdict
import random
import time
import cifar10
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
# from tqdm import *
from scipy.misc import imread
from PIL import Image
from sklearn.neural_network import MLPClassifier
from functools import wraps
from time import time as _timenow
from sys import stderr
from sklearn import svm
from sklearn.linear_model import LogisticRegression

path = "cifar-10-batches-py"

cifar10.data_path = "."
cifar10.load_class_names()
images_train_X, images_class_X, images_label_X = cifar10.load_training_data()
images_train_Y, images_class_Y, images_label_Y = cifar10.load_test_data()

images_train_X = np.array(images_train_X)

# images_train_X = images_train_X.flatten()
len1 = images_train_X.shape[0]
images_train_X = images_train_X.reshape(len1, -1)

len2 = images_train_Y.shape[0]
images_train_Y = images_train_Y.reshape(len2, -1)

# # 
# # 
# # PCA + svm kernel - 1
# pca = PCA(n_components=0.95,svd_solver ='full')

# X_train	=	pca.fit_transform(images_train_X)
# Y_train	=	pca.fit(images_train_Y)
# print(images_label_X.shape)

# rbf_svc = svm.SVC()
# rbf_svc.fit(images_train_X, images_class_X)
# X_predict = rbf_svc.predict(images_train_Y)

# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# Ans = accuracy_score(images_class_Y,X_predict)
# print(Ans)
# # 
# # 


# # # 
# # # 
# # PCA + linear svm - 2
# PCA + svm kernel
# pca = PCA(n_components=0.95,svd_solver ='full')

# X_train	=	pca.fit_transform(images_train_X)
# Y_train	=	pca.fit(images_train_Y)

# clf = LinearSVC()
# clf.fit(images_train_X,images_class_X)
# X_predict = rbf_svc.predict(images_train_Y)
# Ans = accuracy_score(images_class_Y,X_predict

# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# print(Ans)

#
# 
# PCA + MLP       - 3
# pca = PCA(n_components=0.95,svd_solver ='full')

# X_train	=	pca.fit_transform(images_train_X)
# Y_train	=	pca.fit(images_train_Y)

# MLP_PCA = sklearn.neural_network.MLPClassifier()
# MLP_PCA.fit(images_train_X,images_class_X)
# X_predict = rbf_svc.predict(images_train_Y)
# Ans = accuracy_score(images_class_Y,X_predict)

# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# print(Ans)
# # 
# 

# 

# PCA + LogReg   - 4 

pca = PCA(n_components=0.95, svd_solver='full')

X_train = pca.fit_transform(images_train_X)
Y_train = pca.fit(images_train_Y)

LogReg_PCA = LogisticRegression()
LogReg_PCA.fit(images_train_X, images_class_X)
X_predict = LogReg_PCA.predict(images_train_Y)
Ans = accuracy_score(images_class_Y, X_predict)
Ans_f1 = f1_score(images_class_Y, X_predict, average='micro')
print(Ans_f1)
print(Ans)

# 


# RAW + MLP - 5

# MLP_PCA = sklearn.neural_network.MLPClassifier()
# MLP_PCA.fit(images_train_X,images_class_X)
# X_predict = MLP_PCA.predict(images_train_Y)
# Ans = accuracy_score(images_class_Y,X_predict)

# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# print(Ans)

# 
# 

# 
# 
# RAW + rbf - 6 

# rbf_svc = svm.SVC()
# rbf_svc.fit(images_train_X, images_class_X)
# X_predict = rbf_svc.predict(images_train_Y)

# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# Ans = accuracy_score(images_class_Y,X_predict)
# print(Ans)
#  
# 


# 
# 
# RAW + linear - 7

# clf = LinearSVC()
# clf.fit(images_train_X,images_class_X)
# X_predict = clf.fit.predict(images_train_Y)
# Ans = accuracy_score(images_class_Y,X_predict

# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# print(Ans)
# # 
# 

# 
# 
#  RAW + LogReg - 8

# LogReg_PCA = LogisticRegression()
# LogReg_PCA.fit(images_train_X,images_class_X)
# X_predict = LogReg_PCA.predict(images_train_Y)
# Ans = accuracy_score(images_class_Y,X_predict)
# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# print(Ans)
# # 
# # 

# 

# # LDA + MLP   - 9 (NOT WORKING)
# LDA = LinearDiscriminantAnalysis(n_components = 8)

# X_train	=  LDA.fit_transform(images_train_X)
# Y_train	=  LDA.fit(images_train_Y)
# MLP_PCA = sklearn.neural_network.MLPClassifier()
# MLP_PCA.fit(images_train_X,images_class_X)
# X_predict = MLP_PCA.predict(images_train_Y)
# Ans = accuracy_score(images_class_Y,X_predict)
# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# print(Ans)
# # 
# 


# # 

# # LDA + rbf - 10
# LDA = LinearDiscriminantAnalysis(n_components = 8)

# X_train	=	LDA.fit_transform(images_train_X)
# Y_train	=	LDA.fit(images_train_Y)

# rbf_svc = svm.SVC()
# rbf_svc.fit(images_train_X, images_class_X)
# X_predict = rbf_svc.predict(images_train_Y)
# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# Ans = accuracy_score(images_class_Y,X_predict)
# print(Ans)

# # 


# 
# 

# 
# 
# LDA + linear - 11

# LDA = LinearDiscriminantAnalysis(n_components = 8)
# 
# X_train	=	LDA.fit_transform(images_train_X)
# Y_train	=	LDA.fit(images_train_Y)
# clf = LinearSVC()
# clf.fit(images_train_X,images_class_X)
# X_predict = clf.predict(images_train_Y)
# Ans = accuracy_score(images_class_Y,X_predict

# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# print(Ans)

#

# 
# LDA + LogReg - 12
# LDA = LinearDiscriminantAnalysis(n_components = 8)
# 
# X_train	=	LDA.fit_transform(images_train_X)
# Y_train	=	LDA.fit(images_train_Y)

# LogReg_PCA = LogisticRegression()
# LogReg_PCA.fit(images_train_X,images_class_X)
# X_predict = LogReg_PCA.predict(images_train_Y)
# Ans = accuracy_score(images_class_Y,X_predict)
# Ans_f1 = f1_score(images_class_Y,X_predict,average='micro')
# print(Ans_f1)
# print(Ans)	
# 
# 


# if __name__ == "__main__":


#	main()
