import argparse
import pickle
import numpy as np
from sklearn import decomposition, discriminant_analysis, linear_model, svm, tree, neural_network
from sklearn.metrics import f1_score, accuracy_score
from tqdm import *


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
    trn_data = np.array(trn_data)
    trn_labels = np.array(trn_labels)
    tst_data = np.array(tst_data)
    tst_labels = np.array(tst_labels)
    return (trn_data - trn_data.mean(axis=0)), trn_labels, (tst_data - trn_data.mean(axis=0)), tst_labels

def pca_func(trn_data, tst_data):
    pca = decomposition.PCA(n_components=0.95)
    pca.fit(trn_data)
    return pca.transform(trn_data), pca.transform(tst_data)
def lda_func(trn_data, trn_label, tst_data):
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=9)
    lda.fit(trn_data, trn_label)
    return lda.transform(trn_data), lda.transform(tst_data)

def log_reg(trn_data, trn_label, tst_data, c):
    print('LogReg')
    model = linear_model.LogisticRegression(C=c, solver='lbfgs', multi_class='multinomial')
    model.fit(trn_data, trn_label)
    return model.predict(tst_data)
def soft_lin_svm(trn_data, trn_label, tst_data, c):
    print('Linear Svm')
    model = svm.LinearSVC(C=c)
    model.fit(trn_data, trn_label)
    return model.predict(tst_data)
def dec_tree(trn_data, trn_label, tst_data, c):
    print('DecTree')
    model = tree.DecisionTreeClassifier(max_depth=c)
    model.fit(trn_data, trn_label)
    return model.predict(tst_data)
def mlpclass(trn_data, trn_label, tst_data, alpha_func, eeta_func):
    print('MLP')
    model = neural_network.MLPClassifier(alpha=alpha_func, learning_rate_init=eeta_func, activation='relu', early_stopping=True)
    model.fit(trn_data, trn_label)
    return model.predict(tst_data)
def evaluate(target, pred):
    f1 = f1_score(target, pred, average='micro')
    acc = accuracy_score(target, pred)
    return f1, acc


parser = argparse.ArgumentParser(description='SMAI Mini project 2', allow_abbrev=True)
parser.add_argument('--classifier', '-c', metavar='c', type=str, nargs=1, help='the classifier')
parser.add_argument('--method_of_dim_red', '-m', metavar='m', type=str, nargs=1, help='The way data should be taken')
parser.add_argument('--hyper_parameter(s)', '-hp', metavar='hp', type=float, nargs='+',
                    help='1 hp for all classifiers except MLP which has 2 hps')
args = parser.parse_args()
arg_dict = vars(args)
print(arg_dict)
method_dim_red = arg_dict['method_of_dim_red'][0]
method_classify = arg_dict['classifier'][0]
hps = arg_dict['hyper_parameter(s)']
print(method_dim_red, method_classify, hps)

# LOAD
cifar_trn_data, cifar_trn_labels, cifar_tst_data, cifar_tst_labels = load_cifar()

# REDUCE DIMENSIONS
if method_dim_red == 'PCA' or method_dim_red == 'pca':
    print('pca')
    reduced_trn_data, reduced_tst_data = pca_func(cifar_trn_data, cifar_tst_data)
elif method_dim_red == 'LDA' or method_dim_red == 'lda':
    print('lda')
    reduced_trn_data, reduced_tst_data = lda_func(cifar_trn_data, cifar_trn_labels, cifar_tst_data)
else:
    print('raw')
    reduced_trn_data, reduced_tst_data = cifar_trn_data, cifar_tst_data
print(reduced_trn_data.shape, reduced_tst_data.shape)

# Predict and print accuracy
if method_classify == 'LogReg':
    predicted = log_reg(reduced_trn_data, cifar_trn_labels, reduced_tst_data, hps[0])
if method_classify == 'LinearSVM':
    predicted = soft_lin_svm(reduced_trn_data, cifar_trn_labels, reduced_tst_data, hps[0])
if method_classify == 'DecTree':
    predicted = dec_tree(reduced_trn_data, cifar_trn_labels, reduced_tst_data, hps[0])
if method_classify == 'MLP':
    predicted = mlpclass(reduced_trn_data, cifar_trn_labels, reduced_tst_data, hps[0], hps[1])
print(evaluate(cifar_tst_labels, predicted))

