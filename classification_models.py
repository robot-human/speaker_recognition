import numpy as np
from math import sqrt
from random import randrange
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GaussianMixture
from env_variables import MODEL_ATTR

MODEL_ATTR["VQ"]["N_CODEWORDS"]
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

def get_best_matching_unit(feat, codebook):
    min_distortion = 1000
    for enum, codeword in enumerate(codebook):
        #distortion = ratio_distortion(np.array(feat), np.array(codeword), R)
        distortion = euclidean_distance(np.array(feat), np.array(codeword))
        if(distortion < min_distortion):
            min_distortion = distortion
            class_num = enum
    return class_num, min_distortion

def assign_classes(features, codebook):
    classes = []
    for feat in features:
        class_num, dist = get_best_matching_unit(feat, codebook)
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

def train_codebook(features, classes, codebook, epochs):
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
    codebook, classes = train_codebook(features, classes, codebook, epochs)
    return codebook, classes

def codebook_distortion(features, codebook):
    #R = np.corrcoef(features.T)
    classes = assign_classes(features, codebook)
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

def run_VQ_model(speaker_ids, features):
    good_classifications = 0
    bad_classifications = 0
    classifications = []
    speaker_models = []
    for enum, id in enumerate(speaker_ids):
        codebook, classes = vector_quantization_trainning(features[id]['train'], MODEL_ATTR["VQ"]["N_CODEWORDS"], MODEL_ATTR["VQ"]["EPOCHS"])
        speaker_models.append(codebook)
    
    for speaker_enum, id in enumerate(speaker_ids):
        for vector in features[id]['test']:
            speaker = -1
            dist = 1/0.00000001
            for enum, speaker_model in enumerate(speaker_models):
                classes = assign_classes(vector, speaker_model)
                speaker_dist = featureset_distortion(vector, classes, speaker_model)
                if(speaker_dist < dist):
                    dist = speaker_dist
                    speaker = enum
            classifications.append(speaker)
            print(id, speaker_ids[speaker])
            if(speaker_enum == speaker):
                good_classifications += 1
            else:
                bad_classifications += 1
    print(f'Casos bien clasificados: {good_classifications}')
    print(f'Casos mal clasificados: {bad_classifications}')
    print("")
    return classifications
######################################################################################################
# Gaussian Mixture Model
def run_GMM_model(speaker_ids, features, scaler):
    good_classifications = 0
    bad_classifications = 0
    classifications = []
    scaled_separate_set = []
    for id in speaker_ids:
        scaled_separate_set.append(scaler.transform(features[id]['train']))

    speaker_gm_models = []
    for sp in scaled_separate_set:
        gm = GaussianMixture(n_components=MODEL_ATTR["GMM"]["N_MIXTURES"], random_state=0, max_iter=MODEL_ATTR["GMM"]["EPOCHS"], tol=1e-8).fit(sp)
        speaker_gm_models.append(gm)

    for speaker_enum, id in enumerate(speaker_ids):
        for vector in features[id]['test']:
            dist = -1/0.000000001
            speaker = -1
            test_data = scaler.transform(vector)
            for enum, model in enumerate(speaker_gm_models):
                speaker_dist = model.score(test_data)
                if(speaker_dist > dist):
                    dist = speaker_dist
                    speaker = enum
            classifications.append(speaker)
            print(id, speaker_ids[speaker])
            if(speaker_enum == speaker):
                good_classifications += 1
            else:
                bad_classifications += 1
    print(f'Casos bien clasificados: {good_classifications}')
    print(f'Casos mal clasificados: {bad_classifications}')
    print("")
    return classifications


######################################################################################################
def run_SVM_model(speaker_ids, features, scaled_train, classes, scaler):
    good_classifications = 0
    bad_classifications = 0
    classifications = []
    #model_svm = SVC(kernel='rbf', max_iter=5000, tol=1e-5,class_weight='balanced')
    #model_svm = LinearSVC(random_state=0, tol=1e-5)
    model_svm = SGDClassifier(max_iter=MODEL_ATTR["SVM"]["EPOCHS"], tol=1e-3)
    model_svm.fit(scaled_train,classes)

    for speaker_enum, id in enumerate(speaker_ids):
        for vector in features[id]['test']:
            test_data = scaler.transform(vector)
            test_classes = model_svm.predict(test_data)
            counts = np.bincount(test_classes)
            speaker = np.argmax(counts)
            classifications.append(speaker)
            print(id, speaker_ids[speaker])
            if(speaker_enum == speaker):
                good_classifications += 1
            else:
                bad_classifications += 1
    print(f'Casos bien clasificados: {good_classifications}')
    print(f'Casos mal clasificados: {bad_classifications}')
    print("")
    return classifications