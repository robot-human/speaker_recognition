import numpy as np
from math import sqrt
from random import randrange
from sklearn import svm

######################################################################################################
# Vector Quantization
def random_codebook(train):
    n_records = len(train)
    n_features = len(train[0])
    codebook = [train[randrange(n_records)][i] for i in range(n_features)]
    return codebook

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def ratio_distortion(a,b,R):
    num = np.dot(np.dot(b.T,R),b)
    den = np.dot(np.dot(a.T,R),a)
    dist = (num/den) -1
    return abs(dist)

def get_best_matching_unit(feat, codebook, R):
    min_distortion = 1000
    for enum, codeword in enumerate(codebook):
        #distortion = ratio_distortion(np.array(feat), np.array(codeword), R)
        distortion = euclidean_distance(np.array(feat), np.array(codeword))
        if(distortion < min_distortion):
            min_distortion = distortion
            class_num = enum
    return class_num, min_distortion

def assign_classes(features, codebook, R=[0]):
    classes = []
    for feat in features:
        class_num, dist = get_best_matching_unit(feat, codebook, R)
        classes.append(class_num)
    return classes

def update_codeword(features, classes, codeword, class_num):
    avg = np.zeros(len(codeword))
    count = 0.000000001
    for enum, feat in enumerate(features):
        if(classes[enum] == class_num):
            avg = avg + feat
            count += 1
    return avg/count

def featureset_distortion(features, classes, codebook):
    distortion = 0
    for enum, feat in enumerate(features):
        distortion = distortion + euclidean_distance(np.array(feat), np.array(codebook[classes[enum]]))
    return distortion

def train_codebook(features, classes, codebook, epochs,  R=[0]):
    for e in range(epochs):
        for i in range(len(codebook)):
            codebook[i] = update_codeword(features, classes, codebook[i], i)
        classes = assign_classes(features, codebook)
        distortion = featureset_distortion(features, classes, codebook)
        #print(f'epoch: {e}, distortion: {distortion}')
    return codebook, classes

def vector_quantization_trainning(features, n_codewords, epochs):
    #R = np.corrcoef(features.T)
    codebook = [random_codebook(features) for i in range(n_codewords)]
    classes = assign_classes(features, codebook)
    codebook, classes = train_codebook(features, classes, codebook, R, epochs)
    return codebook, classes

def codebook_distortion(features, codebook):
    R = np.corrcoef(features.T)
    classes = assign_classes(features, codebook, R)
    distortion = featureset_distortion(features, classes, codebook)
    return distortion

def select_speaker(features, speaker_codebooks):
    min_distortion = 500
    speaker_num = 0
    for enum, codebook in enumerate(speaker_codebooks):
        dist = codebook_distortion(features, codebook)
        if(dist < min_distortion):
            min_distortion = dist
            speaker_num = enum
    return speaker_num, min_distortion
    
######################################################################################################
# Gaussian Mixture Model




######################################################################################################
# Support Vector Machine
kern   = 'poly'
C      = 1
gamma  = 0.1
tol    = 0.001
degree = 3
coef0  = 0
svm_param_list   = [C, gamma, tol, degree, coef0]
model = svm.SVC(kernel=kern, C=svm_param_list[0], gamma='auto', tol=svm_param_list[2], degree=svm_param_list[3], coef0=svm_param_list[4], max_iter=50000)
#model.fit(train_set,train_y)