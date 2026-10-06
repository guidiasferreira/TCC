import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
import time
from igraph import *
import statsmodels.api as sm
import random
import shelve
from sklearn.linear_model import Ridge
import sklearn.metrics as mtr
import sklearn.manifold as mf
from sklearn.decomposition import PCA
import sklearn.preprocessing as pp
import sklearn.metrics.pairwise as mtc
from matplotlib import cm
from keras import backend as K
from PIL import Image
from os import listdir
from os.path import isfile, join
from enum import IntEnum
from sklearn.svm import SVR
import heapq
import pandas as pd
import csv
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
np.set_printoptions(suppress=True)
np.seterr(divide="ignore", invalid="ignore")
np.set_printoptions(threshold=np.inf)


class base(IntEnum):
    Wine = 0
    Iris = 1
    WineQuality = 2
    FGNet = 3
    Concrete = 4
    Fyn = 5
    Fyn2 = 6
    WaterQuality = 7
    SmartCities = 8
    Hoffsteter = 9
    IDH = 10
    PISA = 11
    COVID = 12
    NPK = 13
    Car = 14
    PlantVillage = 15


def setBase(op=0):
    if op == int(base.Wine):
        filename = r"E:\redes complexas\wine.data"
        raw_data = open(filename, "rt")
        data = np.loadtxt(raw_data, delimiter=",")
        data = np.asmatrix(data)
        Y = np.asmatrix(data[:, 0])
        X = data[:, 1:]
        X = np.asarray(X)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.Iris):
        filename = "irisNumerado.data"
        raw_data = open(filename, "rt")
        data = np.loadtxt(raw_data, delimiter=",")
        # alterado
        dataset1 = np.asmatrix(data)
        ##    dimensao = data.shape
        ##    X = data[:,:dimensao[1]-1]
        ##    Y = data[:,dimensao[1]-1]
        ##    return X, Y
        return dataset1

    elif op == int(base.WineQuality):
        filedir = "C:\\Users\\Renan\\Downloads\\redes complexas\\"
        filename = r"C:\Users\Renan\Documents\pastaCompartilhada\Redes Neurais Artificiais\winequality-white.csv"
        raw_data = open(filename, "rt")
        data = np.loadtxt(raw_data, delimiter=";")
        data = np.asmatrix(data)
        Y = np.asmatrix(data[:, (data.A.shape[1] - 1)])
        # Y = np.sqrt(Y)
        X = data[:, 0 : (data.A.shape[1] - 1)]
        X = np.asarray(X)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.IDH):
        filename = r"C:\Users\Renan\Downloads\redes complexas\IDH2013.csv"
        raw_data = open(filename, "rt")
        data = np.loadtxt(raw_data, delimiter=";")
        data = np.asmatrix(data)
        Y = np.asmatrix(data[:, (data.A.shape[1] - 1)])
        # Y = np.sqrt(Y)
        X = data[:, 0 : (data.A.shape[1] - 1)]
        X = np.asarray(X)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.Car):
        filename = r"E:\Desktop\FATEC\Aprendizado de Máquinas\12026\car.data"
        raw_data = open(filename, "rt")
        data = np.loadtxt(raw_data, delimiter=",")
        data = np.asmatrix(data)
        Y = np.asmatrix(data[:, (data.A.shape[1] - 1)])
        # Y = np.sqrt(Y)
        X = data[:, 0 : (data.A.shape[1] - 1)]
        X = np.asarray(X)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.NPK):
        filename = r"E:\baseNPK.csv"
        raw_data = open(filename, "rt")
        data = np.loadtxt(raw_data, delimiter=";")
        data = np.asmatrix(data)
        ##print(data.shape)
        Y = np.asmatrix(data[:, 2:])
        # Y = np.sqrt(Y)
        X = data[:, :2]
        X = np.asarray(X)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.PISA):
        filename = r"C:\Users\Renan\Downloads\redes complexas\PISA2015.csv"
        raw_data = open(filename, "rt")
        data = np.loadtxt(raw_data, delimiter=";")
        data = np.asmatrix(data)
        Y = np.asmatrix(data[:, (data.A.shape[1] - 1)])
        # Y = np.sqrt(Y)
        X = data[:, 0 : (data.A.shape[1] - 1)]
        X = np.asarray(X)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.COVID):
        filename = r"C:\Users\Renan\Downloads\redes complexas\Covid072020100.csv"
        raw_data = open(filename, "rt")
        data = np.loadtxt(raw_data, delimiter=";")
        data = np.asmatrix(data)
        Y = np.asmatrix(data[:, (data.A.shape[1] - 1)])
        # Y = np.sqrt(Y)
        X = data[:, 0 : (data.A.shape[1] - 1)]
        X = np.asarray(X)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.FGNet):
        # input image dimensions
        img_rows, img_cols = 28, 28
        np.set_printoptions(suppress=True)
        np.seterr(divide="ignore", invalid="ignore")
        ##train images
        X = []
        ##train labels
        Y = []
        mypath = "C:\\Users\\Renan\\Documents\\pastaCompartilhada\\Projeto\\algoritmos\\FGNET\\images\\"
        filename = "C:\\Users\\Renan\\Downloads\\redes complexas\\"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        for j in range(len(onlyfiles)):
            im = Image.open(mypath + onlyfiles[j]).convert("L")
            im = im.resize((img_rows, img_cols), Image.ANTIALIAS)
            imv = np.array(im.getdata(), np.uint8).reshape(im.size[1], im.size[0])
            X.append(imv)
            strAge = str(onlyfiles[j])
            try:
                Age = int(strAge[-6:-4])
            except:
                Age = int(strAge[-7:-5])
            Y.append(Age)
        X = np.array(X)
        Y = np.array(Y)

        if K.image_data_format() == "channels_first":
            x = X.reshape(X.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x = X.reshape(X.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x = x.astype("float32")
        x /= 255
        data = []

        for i in range(len(X)):
            x = np.asarray(X[i, :]).flatten()
            data.append(x)

        data = np.asmatrix(data)
        ##Y=np.sqrt(Y)
        Y = np.asmatrix(Y).T
        X = data
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.PlantVillage):
        # input image dimensions
        img_rows, img_cols = 28, 28
        np.set_printoptions(suppress=True)
        np.seterr(divide="ignore", invalid="ignore")
        ##train images
        X = []
        ##train labels
        Y = []
        mypath = "C:\\Users\\Renan\\Documents\\pastaCompartilhada\\Projeto\\algoritmos\\FGNET\\images\\"
        filename = "C:\\Users\\Renan\\Downloads\\redes complexas\\"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        for j in range(len(onlyfiles)):
            im = Image.open(mypath + onlyfiles[j]).convert("L")
            im = im.resize((img_rows, img_cols), Image.ANTIALIAS)
            imv = np.array(im.getdata(), np.uint8).reshape(im.size[1], im.size[0])
            X.append(imv)
            strAge = str(onlyfiles[j])
            try:
                Age = int(strAge[-6:-4])
            except:
                Age = int(strAge[-7:-5])
            Y.append(Age)
        X = np.array(X)
        Y = np.array(Y)

        if K.image_data_format() == "channels_first":
            x = X.reshape(X.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x = X.reshape(X.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x = x.astype("float32")
        x /= 255
        data = []

        for i in range(len(X)):
            x = np.asarray(X[i, :]).flatten()
            data.append(x)

        data = np.asmatrix(data)
        ##Y=np.sqrt(Y)
        Y = np.asmatrix(Y).T
        X = data
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.Concrete):
        filename = r"C:\Users\Renan\Documents\pastaCompartilhada\Projeto\algoritmos\Concrete_Data.CSV"
        raw_data = open(filename, "rt")
        data = np.loadtxt(raw_data, delimiter=";")

        data = np.asmatrix(data)
        Y = np.asmatrix(data[:, (data.A.shape[1] - 1)])
        ##Y=Y/100

        X = data[:, 0 : (data.A.shape[1] - 1)]
        X = np.asarray(X)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.Fyn2):
        mypath = r"C:\Users\Renan\Downloads\FYNR.csv"
        filename = "C:\\Users\\Renan\\Downloads\\redes complexas\\"
        data = pd.read_csv(mypath, delimiter=";")
        data = data.values

        mols = data[:, 0]
        lenMols = np.zeros(len(mols))
        for i in range(len(mols)):
            lenMols[i] = len(str(mols[i]))
        dataset1 = np.zeros((len(mols), int(max(lenMols))))
        for i in range(len(mols)):
            st = str(mols[i])
            for j in range(len(st)):
                dataset1[i, j] = int(ord(st[j]))

        dimC = data.shape[1]
        X = dataset1.copy()
        Y = data[:, dimC - 1 : dimC]

        # Y=Y/max(Y)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.WaterQuality):
        mypath = r"C:\Users\renan\Downloads\redes complexas\waterquality2014comLL.csv"
        filename = "C:\\Users\\Renan\\Downloads\\redes complexas\\"
        raw_data = open(mypath, "rt")
        data = np.loadtxt(raw_data, delimiter=";")

        data = np.asmatrix(data)
        Y = np.asmatrix(data[:, data.A.shape[1] - 1])
        Y = np.log10(Y) / np.log10(6)

        X = data[:, 1 : data.A.shape[1] - 1]
        X = np.asarray(X)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.SmartCities):
        mypath = r"C:\Users\renan\Downloads\redes complexas\smartcitiescomLL.csv"
        filename = "C:\\Users\\Renan\\Downloads\\redes complexas\\"
        raw_data = open(mypath, "rt")
        data = np.loadtxt(raw_data, delimiter=";")

        data = np.asmatrix(data)
        Y = np.asmatrix(data[:, data.A.shape[1] - 1])
        ##Y=np.log10(Y)/np.log10(6)

        X = data[:, 1 : data.A.shape[1] - 1]
        X = np.asarray(X)

        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.Hoffsteter):
        mypath = r"E:\redes complexas\hofftetercomLL.csv"
        filename = "E:\\Downloads\\redes complexas\\"
        raw_data = open(mypath, "rt")
        data = np.loadtxt(raw_data, delimiter=";")

        data = np.asmatrix(data)
        Y = np.asmatrix(data[:, data.A.shape[1] - 1])
        ##Y=np.log10(Y)/np.log10(6)

        X = data[:, 1 : data.A.shape[1] - 1]
        X = np.asarray(X)
        dataset1 = np.hstack((X, Y))
        dataset1 = np.asmatrix(dataset1)
        return dataset1

    elif op == int(base.Fyn):
        mypath = r"C:\Users\Renan\Downloads\FYNR.csv"
        filename = "C:\\Users\\Renan\\Downloads\\redes complexas\\"
        data = pd.read_csv(mypath, delimiter=";")
        data = data.values

        ##        mols=data[:,0]
        ##        lenMols=np.zeros(len(mols))
        ##        for i in range(len(mols)):
        ##            lenMols[i]=len(str(mols[i]))
        ##        dataset1 = np.zeros((len(mols),int(max(lenMols))))
        ##        for i in range(len(mols)):
        ##            st=str(mols[i])
        ##            for j in range(len(st)):
        ##                dataset1[i,j]=int(ord(st[j]))

        sentences = data[:, 0:2]

        wordfreq = {}
        # Verifico se a palavra existe no dicionário e anoto a frequência
        for words in sentences:
            for token in words:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1

        # verifico os mais frequentes tokens
        most_freq = heapq.nlargest(100, wordfreq, key=wordfreq.get)

        # verificando se a palavra existe nas mais frequentes
        sentence_vectors = []
        for token in sentences:
            sent_vec = []
            for word in most_freq:
                if word in token:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            sentence_vectors.append(sent_vec)

        sentence_vectors = np.asarray(sentence_vectors)
        # ----------------------------------TF-IDF--------------------------------------

        # ----------IDF----------
        word_idf_values = {}
        for token in most_freq:
            doc_containing_word = 0
            for words in sentences:
                if token in words:
                    doc_containing_word += 1
            word_idf_values[token] = np.log(len(sentences) / (1 + doc_containing_word))

        # ----------TF----------
        word_tf_values = {}
        for token in most_freq:
            sent_tf_vector = []
            for words in sentences:
                doc_freq = 0
                for word in words:
                    if token == word:
                        doc_freq += 1
                word_tf = doc_freq / len(words)
                sent_tf_vector.append(word_tf)
            word_tf_values[token] = sent_tf_vector

        # --------TF-IDF--------
        tfidf_values = []
        for token in word_tf_values.keys():
            tfidf_sentences = []
            for tf_sentence in word_tf_values[token]:
                tf_idf_score = tf_sentence * word_idf_values[token]
                tfidf_sentences.append(tf_idf_score)
            tfidf_values.append(tfidf_sentences)

        tf_idf_model = np.asarray(tfidf_values)

        tf_idf_model = np.transpose(tf_idf_model)

        dataset1 = np.hstack((tf_idf_model, data[:, 5:7]))
        dataset1 = np.asmatrix(dataset1)

        dim = dataset1.A.shape[1]
        X = dataset1[:, 0 : dim - 1]
        Y = dataset1[:, dim - 1 : dim]

        # Y=Y/max(Y)
        dataset1 = np.hstack((X, Y))
        return dataset1


##---------------------------escolhendo a base -------------------------------------
dataset = setBase(op=base.PlantVillage)
diml = dataset.A.shape[0]
dimc = dataset.A.shape[1]
X = dataset[:, 0 : (dimc - 1)]
Y = dataset[:, (dimc - 1) :]


class reduDim(IntEnum):
    RAW = 0
    TSNE = 1
    PCA = 2
    LLE = 3
    SE = 4
    MDS = 5


def setReduDim(data=dataset, op=1):
    if op == int(reduDim.PCA):
        data = np.asarray(data)
        pca1 = PCA(n_components=3)
        X_embedded = pca1.fit_transform(data)

        return X_embedded

    elif op == int(reduDim.TSNE):
        X_embedded = mf.TSNE(n_components=3).fit_transform(data)
        return X_embedded

    elif op == int(reduDim.LLE):
        X_embedded = mf.LocallyLinearEmbedding(n_components=2).fit_transform(data)
        return X_embedded

    elif op == int(reduDim.SE):
        X_embedded = mf.SpectralEmbedding(n_components=2).fit_transform(data)
        return X_embedded

    elif op == int(reduDim.MDS):
        X_embedded = mf.MDS(n_components=2).fit_transform(data)
        return X_embedded

    else:
        X_embedded = data
        return X_embedded


def kmeans(dataset=dataset, k=2, threshold=0.001):
    dataset = np.asmatrix(dataset)
    ##dataset= pp.StandardScaler().fit_transform(dataset)
    ids = np.random.randint(0, len(dataset), k)
    centers = []
    for i in range(0, k):
        centers.append(dataset[int(ids[i]), :])
    centers = np.asmatrix(np.squeeze(centers))
    closest = np.zeros(len(dataset))
    W = np.zeros(len(dataset))

    div = threshold * 2
    while div > threshold:
        for i in range(len(dataset)):
            row = dataset[i, :]
            euclidean = np.zeros(len(ids))
            for j in range(len(ids)):
                euclidean[j] = np.sqrt(np.sum(np.power((row - centers[j, :]), 2)))
            closest[i] = np.argmin(euclidean)
            W[i] = np.min(euclidean)
        old = centers.copy()
        for i in range(k):
            new = []
            for j in range(len(dataset)):
                if closest[j] == i:
                    new.append(dataset[j, :])
            new = np.asmatrix(np.squeeze(new))
            centers[i, :] = np.mean(new, axis=0)

        div = np.sqrt(np.sum(np.power((old - centers), 2)))
    W = np.power(W, 2)
    W = np.sum(W)
    return centers, W, closest


def hartigan(dataset, eta=10):
    i = 2
    H = eta * 2
    w = []
    w1 = []

    centers, W, closest = kmeans(dataset=dataset, k=i, threshold=0.001)
    w = np.round(W.copy(), 2)
    gamma = len(dataset) - i - 1

    while H > eta:
        try:
            centers, W, closest = kmeans(dataset=dataset, k=(i + 1), threshold=0.001)
            w1 = np.round(W.copy(), 2)
        except:
            i
        H = np.round(gamma * ((w - w1) / (w1)), 2)
        i = i + 1
        w = w1.copy()
        gamma = len(dataset) - i - 1

    idC = i - 1

    return H, idC


class pre(IntEnum):
    StdScaler = 0
    MinMax = 1
    QuantileTransform = 2
    MaxAbsScale = 3
    PowerTransform = 4
    Scale = 5
    Normalize = 6
    RobustScale = 7
    PolinomialFeatures = 8
    FunctionTransformer = 9

    ##KernelCenterer=8
    ##OneHotEncoder=10
    ##OrdinalEncoder=11
    ##KbinsDiscretizer=12


class metricaD(IntEnum):
    Eucliana = 0
    Cosseno = 1
    Hamming = 2
    Manhattan = 3
    Chebyshev = 4
    Minkowski = 5
    Jaccard = 6
    Haversine = 7
    Sorensen = 8
    Klaplaciano = 9
    Klinear = 10
    Kdistancia = 11
    Krbf = 12
    Ksigmoid = 13
    Kpolinomial = 14


def preProcessar(dataset=dataset, op=0):

    if op == int(pre.StdScaler):
        dataset1 = pp.StandardScaler().fit_transform(np.asarray(dataset))
        return dataset1
    elif op == int(pre.MinMax):
        dataset1 = pp.MinMaxScaler().fit_transform(np.asarray(dataset))
        return dataset1
    elif op == int(pre.QuantileTransform):
        dataset1 = pp.QuantileTransformer(n_quantiles=10, random_state=0).fit_transform(
            np.asarray(dataset)
        )
        return dataset1
    elif op == int(pre.MaxAbsScale):
        dataset1 = pp.MaxAbsScaler().fit_transform(np.asarray(dataset))
        return dataset1
    elif op == int(pre.PowerTransform):
        dataset1 = pp.PowerTransformer(
            method="yeo-johnson", standardize=False
        ).fit_transform(np.asarray(dataset))
        return dataset1
    elif op == int(pre.Scale):
        dataset1 = pp.scale(np.asarray(dataset))
        return dataset1
    elif op == int(pre.Normalize):
        dataset1 = pp.normalize(np.asarray(dataset), norm="l2")
        return dataset1
    elif op == int(pre.RobustScale):
        dataset1 = pp.RobustScaler().fit_transform(np.asarray(dataset))
        return dataset1
    ##elif op == pre.KernelCenterer:
    ##    dataset1 = pp.KernelCenterer().fit_transform(dataset)
    ##    return dataset1
    elif op == int(pre.PolinomialFeatures):
        dataset1 = pp.PolynomialFeatures().fit_transform(np.asarray(dataset))
        return dataset1
    ##elif op == pre.OneHotEncoder:
    ##elif op == pre.OrdinalEncoder:
    ##elif op == pre.KbinsDiscretizer:
    ##    dataset1 = pp.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit_transform(dataset)
    ##    return dataset1
    elif op == int(pre.FunctionTransformer):
        dataset1 = pp.FunctionTransformer(np.log1p, validate=True).fit_transform(
            np.asarray(dataset)
        )
        return dataset1


def testTransform(dataset=dataset):
    enum_list = list(map(int, pre))
    h = np.zeros(len(enum_list))
    cl = h.copy()
    for i in range(len(enum_list)):
        dt1 = preProcessar(dataset=dataset, op=i)
        H, idC = hartigan(dataset=dt1, eta=10)
        h[i] = H
        cl[i] = idC
        print("hartigam =" + str(i + 1) + " de " + str(len(enum_list)))
    return h, cl


def plotGraphClustersScaleReduction(k=2, op=0, red=0):
    model = Model()
    model.Xtrain = X.copy()
    model.Ytrain = Y.copy()

    print("Aplicando Escala")
    model.Xtrain = preProcessar(dataset=model.Xtrain, op=op)

    print("Aplicando Redução de dimensionalidade")
    model.Xtrain = setReduDim(data=model.Xtrain, op=red)

    print("Buscando distancias")
    D = distance(model.Xtrain)

    print("Aplicando KNN")
    knnMat = knn(D, 10)

    ##    for i in range(len(knnMat)):
    ##        for j in range(knnMat.shape[1]):

    print("Aplicando RBF")
    Wi = buildW(D, knnMat, sigma=0.5)
    ##W=buildW(model.Xtrain,sigma=sigma)

    centers, W, closest = kmeans(dataset=dataset, k=k, threshold=0.001)

    print("Gerando Grafo")
    model.g = Graph()
    model.g.add_vertices(len(Wi))
    model.g.vs["label"] = range(0, len(Wi))

    model.g.vs["classes"] = closest
    ##color_dict = {0: "grey", 1: "blue", 2: "red", 3: "green"}

    print("Ajustando paleta rainbow")
    pal = RainbowPalette(n=255)
    ##matiz=254/max(model.Ytrain)
    d = max(closest) - min(closest)
    matiz = (254 - 31.75) / d

    print("Aplicando Paleta")
    for i in range(len(model.Ytrain)):
        model.g.vs[i]["color"] = pal.get(
            int(((closest[i] - min(closest)) * matiz) + 31.75)
        )

    ##switch
    print("Gerando arestas")
    for i in range(0, Wi.A.shape[0]):
        for j in range(0, Wi.A.shape[1]):
            if Wi[i, j] > 0 and knnMat[i, j] != -1:
                if model.g.are_connected(i, int(knnMat[i, j])) == False:
                    model.g.add_edge(i, int(knnMat[i, j]))
                    model.g.es[model.g.get_eid(i, int(knnMat[i, j]))]["weight"] = Wi[
                        i, j
                    ]

    print("Plot grafo rotulado")
    random.seed(140)
    out = plot(model.g, layout="kk", vertex_size=25)
    # out.save(filename + 'GWineTSNERandonWalk' + 'sigma' + str(sigma) + 'r' + str(r) + 'k' + str(K) + 'g1' + str(
    #     tipo) + '.png')


def plotGraphScaleReduction(op=0, red=0):
    model = Model()
    model.Xtrain = X.copy()
    model.Ytrain = Y.copy()
    print("Aplicando Escala")
    model.Xtrain = preProcessar(dataset=model.Xtrain, op=op)

    print("Aplicando Redução de dimensionalidade")
    model.Xtrain = setReduDim(data=model.Xtrain, op=red)

    print("Buscando distancias")
    D = distance(model.Xtrain)

    print("Aplicando KNN")
    knnMat = knn(D, 10)

    ##    for i in range(len(knnMat)):
    ##        for j in range(knnMat.shape[1]):

    print("Aplicando RBF")
    Wi = buildW(D, knnMat, sigma=0.5)
    ##W=buildW(model.Xtrain,sigma=sigma)

    print("Gerando Grafo")
    model.g = Graph()
    model.g.add_vertices(len(Wi))
    model.g.vs["label"] = range(0, len(Wi))

    model.g.vs["classes"] = model.Ytrain
    ##color_dict = {0: "grey", 1: "blue", 2: "red", 3: "green"}

    print("Ajustando paleta rainbow")
    pal = RainbowPalette(n=255)
    ##matiz=254/max(model.Ytrain)
    d = max(model.Ytrain) - min(model.Ytrain)
    matiz = (254 - 31.75) / d

    print("Aplicando Paleta")
    for i in range(len(model.Ytrain)):
        model.g.vs[i]["color"] = pal.get(
            int(((model.Ytrain[i] - min(model.Ytrain)) * matiz) + 31.75)
        )

    ##switch
    print("Gerando arestas")
    for i in range(0, Wi.A.shape[0]):
        for j in range(0, Wi.A.shape[1]):
            if Wi[i, j] > 0 and knnMat[i, j] != -1:
                if model.g.are_connected(i, int(knnMat[i, j])) == False:
                    model.g.add_edge(i, int(knnMat[i, j]))
                    model.g.es[model.g.get_eid(i, int(knnMat[i, j]))]["weight"] = Wi[
                        i, j
                    ]

    print("Plot grafo rotulado")
    random.seed(140)
    out = plot(model.g, layout="kk", vertex_size=25)
    # out.save(filename + 'GWineTSNERandonWalk' + 'sigma' + str(sigma) + 'r' + str(r) + 'k' + str(K) + 'g1' + str(
    #     tipo) + '.png')


def plotGraphScaleReduction(op=0, red=0):
    model = Model()
    model.Xtrain = X.copy()
    model.Ytrain = Y.copy()
    print("Aplicando Escala")
    model.Xtrain = preProcessar(dataset=model.Xtrain, op=op)

    print("Aplicando Redução de dimensionalidade")
    model.Xtrain = setReduDim(data=model.Xtrain, op=red)

    print("Buscando distancias")
    D = distance(model.Xtrain)

    print("Aplicando KNN")
    knnMat = knn(D, 10)

    ##    for i in range(len(knnMat)):
    ##        for j in range(knnMat.shape[1]):

    print("Aplicando RBF")
    Wi = buildW(D, knnMat, sigma=0.5)
    ##W=buildW(model.Xtrain,sigma=sigma)

    print("Gerando Grafo")
    model.g = Graph()
    model.g.add_vertices(len(Wi))
    model.g.vs["label"] = range(0, len(Wi))

    model.g.vs["classes"] = model.Ytrain
    ##color_dict = {0: "grey", 1: "blue", 2: "red", 3: "green"}

    print("Ajustando paleta rainbow")
    pal = RainbowPalette(n=255)
    ##matiz=254/max(model.Ytrain)
    d = max(model.Ytrain) - min(model.Ytrain)
    matiz = (254 - 31.75) / d

    print("Aplicando Paleta")
    for i in range(len(model.Ytrain)):
        model.g.vs[i]["color"] = pal.get(
            int(((model.Ytrain[i] - min(model.Ytrain)) * matiz) + 31.75)
        )

    ##switch
    print("Gerando arestas")
    for i in range(0, Wi.A.shape[0]):
        for j in range(0, Wi.A.shape[1]):
            if Wi[i, j] > 0 and knnMat[i, j] != -1:
                if model.g.are_connected(i, int(knnMat[i, j])) == False:
                    model.g.add_edge(i, int(knnMat[i, j]))
                    model.g.es[model.g.get_eid(i, int(knnMat[i, j]))]["weight"] = Wi[
                        i, j
                    ]

    print("Plot grafo rotulado")
    random.seed(140)
    out = plot(model.g, layout="kk", vertex_size=25)
    # out.save(filename + 'GWineTSNERandonWalk' + 'sigma' + str(sigma) + 'r' + str(r) + 'k' + str(K) + 'g1' + str(
    #     tipo) + '.png')


def construirGrafoKnn(model, knnMat, D):
    ##    print("Gerando Grafo")
    model.g = Graph()
    model.g.add_vertices(len(knnMat))
    # g.to_undirected)
    ##    model.g.vs["label"] = range(0,len(W))
    model.g.vs["id"] = range(0, len(knnMat))
    model.g1 = model.g.copy()

    model.g1.vs["classes"] = Y
    model.g.vs["label"] = range(0, len(Y))
    ##color_dict = {0: "grey", 1: "blue", 2: "red", 3: "green"}

    ##    print("Ajustando paleta rainbow")
    pal = RainbowPalette(n=255)
    ##matiz=254/max(model.Ytrain)
    d = max(Y) - min(Y)
    matiz = (254 - 31.75) / d

    ##    print("Aplicando Paleta")
    for i in range(len(Y)):
        ##model.g1.vs[i]['color']=pal.get(int(model.Ytrain[i]*matiz))
        model.g1.vs[i]["color"] = pal.get(int(((Y[i] - min(Y)) * matiz) + 31.75))

        if Y[i] == 0.0:
            model.g.vs[i]["color"] = color_name_to_rgba("grey")
        else:
            ##model.g.vs[i]['color']=pal.get(int(model.YtrainVet[i]*matiz))
            model.g.vs[i]["color"] = pal.get(int(((Y[i] - min(Y)) * matiz) + 31.75))

    ##switch
    ##    print("Gerando arestas")
    for i in range(0, knnMat.shape[0]):
        for j in range(0, knnMat.shape[1]):
            if int(knnMat[i, j]) >= 0:
                if model.g.are_connected(i, int(knnMat[i, j])) == False:
                    model.g.add_edge(i, int(knnMat[i, j]))
                    model.g.es[model.g.get_eid(i, int(knnMat[i, j]))]["weight"] = D[
                        i, j
                    ]
                    model.g1.add_edge(i, int(knnMat[i, j]))
                    model.g1.es[model.g1.get_eid(i, int(knnMat[i, j]))]["weight"] = D[
                        i, j
                    ]
        ##print("i="+str(i))

    return model


def construirGrafoSmallWorld(model, knnMat, W):
    ##    print("Gerando Grafo")
    model.g = Graph()
    model.g.add_vertices(len(W))
    # g.to_undirected)
    ##    model.g.vs["label"] = range(0,len(W))
    model.g.vs["id"] = range(0, len(W))
    model.g.vs["iter"] = 0
    model.g1 = model.g.copy()

    model.g.vs["classes"] = model.YtrainVet
    model.g.vs["iter"] = model.rotulosIni
    model.g1.vs["classes"] = model.Ytrain
    ##color_dict = {0: "grey", 1: "blue", 2: "red", 3: "green"}

    ##    print("Ajustando paleta rainbow")
    pal = RainbowPalette(n=255)
    ##matiz=254/max(model.Ytrain)
    d = max(model.Ytrain) - min(model.Ytrain)
    matiz = (254 - 31.75) / d

    ##    print("Aplicando Paleta")
    for i in range(len(model.YtrainVet)):
        ##model.g1.vs[i]['color']=pal.get(int(model.Ytrain[i]*matiz))
        model.g1.vs[i]["color"] = pal.get(
            int(((model.Ytrain[i] - min(model.Ytrain)) * matiz) + 31.75)
        )

        if model.YtrainVet[i] == 0.0:
            model.g.vs[i]["color"] = color_name_to_rgba("grey")
        else:
            ##model.g.vs[i]['color']=pal.get(int(model.YtrainVet[i]*matiz))
            model.g.vs[i]["color"] = pal.get(
                int(((model.YtrainVet[i] - min(model.Ytrain)) * matiz) + 31.75)
            )

    ##switch
    ##    print("Gerando arestas")
    for i in range(0, W.A.shape[0]):
        for j in range(0, W.A.shape[1]):
            if W[i, j] > 0 and int(knnMat[i, j]) >= 0:
                if model.g.are_connected(i, int(knnMat[i, j])) == False:
                    model.g.add_edge(i, int(knnMat[i, j]))
                    model.g.es[model.g.get_eid(i, int(knnMat[i, j]))]["weight"] = W[
                        i, j
                    ]
                    model.g1.add_edge(i, int(knnMat[i, j]))
                    model.g1.es[model.g1.get_eid(i, int(knnMat[i, j]))]["weight"] = W[
                        i, j
                    ]
        ##print("i="+str(i))
    return model


def construirGrafoEspectral(model, W):
    ##    print("Gerando Grafo")
    model.gRel = Graph()
    model.gRel.add_vertices(len(W))
    # g.to_undirected)
    ##    model.g.vs["label"] = range(0,len(W))
    model.gRel.vs["id"] = range(0, len(W))
    model.gRel.vs["iter"] = 0

    model.gRel.vs["classes"] = Y
    model.gRel.vs["label"] = range(0, len(Y))
    ##color_dict = {0: "grey", 1: "blue", 2: "red", 3: "green"}

    ##    print("Ajustando paleta rainbow")
    pal = RainbowPalette(n=255)
    ##matiz=254/max(model.Ytrain)
    d = max(Y) - min(Y)
    matiz = (254 - 31.75) / d

    ##    print("Aplicando Paleta")
    for i in range(len(Y)):
        ##model.g1.vs[i]['color']=pal.get(int(model.Ytrain[i]*matiz))
        model.gRel.vs[i]["color"] = pal.get(int(((Y[i] - min(Y)) * matiz) + 31.75))

    ##switch
    ##    print("Gerando arestas")
    for i in range(0, W.A.shape[0]):
        for j in range(0, W.A.shape[1]):
            if W[i, j] > 0 and i != j:
                if model.gRel.are_connected(i, j) == False:
                    model.gRel.add_edge(i, j)
                    model.gRel.es[model.gRel.get_eid(i, j)]["weight"] = W[i, j]
    return model


def plotGraphReductionScale(op=0, red=0):
    model = Model()
    model.Xtrain = X.copy()
    model.Ytrain = Y.copy()

    print("Aplicando Redução de dimensionalidade")
    model.Xtrain = setReduDim(data=model.Xtrain, op=red)

    print("Aplicando Escala")
    model.Xtrain = preProcessar(dataset=model.Xtrain, op=op)

    print("Buscando distancias")
    D = distance(model.Xtrain)

    print("Aplicando KNN")
    knnMat = knn(D, 10)

    ##    for i in range(len(knnMat)):
    ##        for j in range(knnMat.shape[1]):

    print("Aplicando RBF")
    Wi = buildW(D, knnMat, sigma=0.5)
    ##W=buildW(model.Xtrain,sigma=sigma)

    print("Gerando Grafo")
    model.g = Graph()
    model.g.add_vertices(len(Wi))
    model.g.vs["label"] = range(0, len(Wi))

    model.g.vs["classes"] = model.Ytrain
    ##color_dict = {0: "grey", 1: "blue", 2: "red", 3: "green"}

    print("Ajustando paleta rainbow")
    pal = RainbowPalette(n=255)
    ##matiz=254/max(model.Ytrain)
    d = max(model.Ytrain) - min(model.Ytrain)
    matiz = (254 - 31.75) / d

    print("Aplicando Paleta")
    for i in range(len(model.Ytrain)):
        model.g.vs[i]["color"] = pal.get(
            int(((model.Ytrain[i] - min(model.Ytrain)) * matiz) + 31.75)
        )

    ##switch
    print("Gerando arestas")
    for i in range(0, Wi.A.shape[0]):
        for j in range(0, Wi.A.shape[1]):
            if Wi[i, j] > 0 and knnMat[i, j] != -1:
                if model.g.are_connected(i, int(knnMat[i, j])) == False:
                    model.g.add_edge(i, int(knnMat[i, j]))
                    model.g.es[model.g.get_eid(i, int(knnMat[i, j]))]["weight"] = Wi[
                        i, j
                    ]

    print("Plot grafo rotulado")
    random.seed(140)
    out = plot(model.g, layout="kk", vertex_size=25)
    # out.save(filename + 'GWineTSNERandonWalk' + 'sigma' + str(sigma) + 'r' + str(r) + 'k' + str(K) + 'g1' + str(
    #     tipo) + '.png')


def writeOutput(var):
    text_file = open("C:\\Users\\renan\\Desktop\\Output.txt", "w")
    text_file.write(str(var))
    text_file.close()


def plotdataClustersReductionScale(k=2, op=0, red=0):
    model = Model()
    model.Xtrain = X.copy()
    model.Ytrain = Y.copy()

    print("Aplicando Redução de dimensionalidade")
    model.Xtrain = setReduDim(data=model.Xtrain, op=red)

    print("Aplicando Escala")
    model.Xtrain = preProcessar(dataset=model.Xtrain, op=op)

    ##    print("Buscando distancias")
    ##    D=distance(model.Xtrain)

    centers, W, closest = kmeans(dataset=model.Xtrain, k=k, threshold=0.001)

    print("Plot rotulado")

    volume = 10

    fig, ax = plt.subplots()
    ax.scatter(model.Xtrain[:, 0], model.Xtrain[:, 1], c=closest, s=volume, alpha=0.5)

    ax.set_xlabel(r"$X$", fontsize=15)
    ax.set_ylabel(r"$Y$", fontsize=15)
    ax.set_title("Dados no $R^{2}$")

    ax.grid(True)
    fig.tight_layout()

    plt.show()


def plotdataClustersScaleReduction(k=2, op=0, red=0):
    model = Model()
    model.Xtrain = X.copy()
    model.Ytrain = Y.copy()

    print("Aplicando Escala")
    model.Xtrain = preProcessar(dataset=model.Xtrain, op=op)

    print("Aplicando Redução de dimensionalidade")
    model.Xtrain = setReduDim(data=model.Xtrain, op=red)

    ##    print("Buscando distancias")
    ##    D=distance(model.Xtrain)

    centers, W, closest = kmeans(dataset=model.Xtrain, k=k, threshold=0.001)

    print("Plot rotulado")

    volume = 10

    fig, ax = plt.subplots()
    ax.scatter(model.Xtrain[:, 0], model.Xtrain[:, 1], c=closest, s=volume, alpha=0.5)

    ax.set_xlabel(r"$X$", fontsize=15)
    ax.set_ylabel(r"$Y$", fontsize=15)
    ax.set_title("Dados no $R^{2}$")

    ax.grid(True)
    fig.tight_layout()

    plt.show()


class Model:
    ##def __init__(self, hidden, output,lenI,lenO,lenH):
    g = []
    g0 = []
    g1 = []
    yL = []
    yTrain = []
    Xtrain = []
    YtrainVet = []
    yRel = []
    S = []
    nb = []
    corr = []

    def print(model):
        print("Grafo propagado=" + str(model.g))
        print("Grafo sem propagação=" + str(model.g0))
        print("Grafo Original=" + str(model.g1))
        print("Vetor Resposta Propagado=" + str(model.yL))
        print("Vetor Resposta Proposicional=" + str(model.yRel))
        print("Vetor Resposta Original=" + str(model.yTrain))
        print("Vetor X Original=" + str(model.Xtrain))
        print("Vetor Resposta sem propagação=" + str(model.YtrainVet))
        print("variancia e erro abs=" + str(model.S))
        print("numero de vizinhos total e ativos=" + str(model.nb))
        print("Vetor de correlação=" + str(model.corr))


def pca(x):
    m = np.mean(x.T, axis=1)
    c = x - m.T
    # v=np.cov(c)
    v = np.cov(c.T)
    values, vectors = np.linalg.eig(v)
    valuespercent = values * 100 / sum(values)
    p = vectors.T.dot(c.T)


def plot3dCl(X_embedded, Y, labels=1):
    fig = plt.figure()
    Y = np.asarray(Y, dtype="int64")
    # ax = fig.gca(projection='3d')
    ax = plt.figure().add_subplot(projection="3d")
    # Plot a sin curve using the x and y axes.
    x = X_embedded[:, 0]
    y = X_embedded[:, 1]
    z = X_embedded[:, 2]

    ##    c_list = []
    ##    d=max(Y)-min(Y)
    ##    matiz=(254-31.75)/d
    ##    for i in range(len(Y)):
    ##        model.g.vs[i]['color'] = pal.get(int(((model.Ytrain[i] - min(model.Ytrain)) * matiz) + 31.75))
    if labels == 1:
        for Ys, xs, ys, zs in zip(Y, x, y, z):
            label = " %.2f " % (Ys)
            ax.text(xs, ys, zs, label, None)

    K = len(Y)
    # colors = cm.hsv(np.arange(K)/float(K))
    colors = Y
    lbl = np.unique(Y)
    ax.scatter(x, y, z, c=colors)
    # ax.scatter(x, y, z, c=colors)
    ax.legend(ncol=lbl)
    minScater = X_embedded.min() - ((10 * X_embedded.min()) / 100)
    maxScater = X_embedded.max() + ((10 * X_embedded.max()) / 100)
    ax.set_xlim(minScater, maxScater)
    ax.set_ylim(minScater, maxScater)
    ax.set_zlim(minScater, maxScater)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20.0, azim=-35)
    plt.show()


def plot3d(X_embedded, Y, labels=1):
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = plt.figure().add_subplot(projection="3d")
    # Plot a sin curve using the x and y axes.
    x = X_embedded[:, 0]
    y = X_embedded[:, 1]
    z = X_embedded[:, 2]

    ##    c_list = []
    ##    d=max(Y)-min(Y)
    ##    matiz=(254-31.75)/d
    ##    for i in range(len(Y)):
    ##        model.g.vs[i]['color'] = pal.get(int(((model.Ytrain[i] - min(model.Ytrain)) * matiz) + 31.75))
    if labels == 1:
        for Ys, xs, ys, zs in zip(Y, x, y, z):
            label = " %.2f " % (Ys)
            ax.text(xs, ys, zs, label, None)

    ##    K=len(Y)
    ##    colors = cm.hsv(np.arange(K)/float(K))
    colors = Y

    ax.scatter(x, y, z, c=colors, label="points in (x, y, z)")
    ax.legend()
    minScater = X_embedded.min() - ((10 * X_embedded.min()) / 100)
    maxScater = X_embedded.max() + ((10 * X_embedded.max()) / 100)
    ax.set_xlim(minScater, maxScater)
    ax.set_ylim(minScater, maxScater)
    ax.set_zlim(minScater, maxScater)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20.0, azim=-35)
    plt.show()


def distance(X):
    X = np.asmatrix(X)

    D = np.asmatrix(np.zeros(shape=(X.A.shape[0], X.A.shape[0])))

    for i in range(X.A.shape[0]):
        ##for j in range(X.A.shape[0]):
        j = 0
        while j < i:
            ##print("i="+str(i)+" j= "+str(j))
            D[i, j] = np.sqrt(np.sum(np.power((X[i, :] - X[j, :]), 2), axis=1))
            D[j, i] = D[i, j]
            j = j + 1
    return D


def processarDistancia(dataset=dataset, op=0):

    ##    Eucliana = 0
    ##    Cosseno = 1
    ##    Hamming = 2
    ##    Manhattan = 3
    ##    Chebyshev=4
    ##    Minkowski=5
    ##    Jaccard=6
    ##    Haversine=7
    ##    Sorensen=8
    ##    Klaplaciano=9
    ##    Klinear=10
    ##    Kdistancia=11
    ##    Krbf=12
    ##    Ksigmoid=13
    ##    Kpolinomial=14

    if op == int(metricaD.Eucliana):
        dataset1 = mtc.euclidean_distances(dataset, dataset)
        return dataset1
    elif op == int(metricaD.Cosseno):
        dataset1 = mtc.cosine_distances(dataset, dataset)
        return dataset1
    elif op == int(metricaD.Chebyshev):
        dataset1 = mtc.cheby(dataset, dataset)
        return dataset1
    elif op == int(metricaD.Haversine):
        dataset1 = pp.MaxAbsScaler().fit_transform(dataset)
        return dataset1
    elif op == int(metricaD.Hamming):
        dataset1 = pp.PowerTransformer(
            method="yeo-johnson", standardize=False
        ).fit_transform(dataset)
        return dataset1
    elif op == int(metricaD.Jaccard):
        dataset1 = pp.scale(dataset)
        return dataset1
    elif op == int(metricaD.Manhattan):
        dataset1 = pp.normalize(dataset, norm="l2")
        return dataset1
    elif op == int(metricaD.Minkowski):
        dataset1 = pp.RobustScaler().fit_transform(dataset)
        return dataset1
    elif op == int(metricaD.Sorensen):
        dataset1 = pp.RobustScaler().fit_transform(dataset)
        return dataset1
    elif op == int(metricaD.Kdistancia):
        dataset1 = pp.RobustScaler().fit_transform(dataset)
        return dataset1
    elif op == int(metricaD.Klaplaciano):
        dataset1 = pp.RobustScaler().fit_transform(dataset)
        return dataset1
    elif op == int(metricaD.Klinear):
        dataset1 = pp.RobustScaler().fit_transform(dataset)
        return dataset1
    elif op == int(metricaD.Krbf):
        dataset1 = pp.RobustScaler().fit_transform(dataset)
        return dataset1
    elif op == int(metricaD.Ksigmoid):
        dataset1 = pp.RobustScaler().fit_transform(dataset)
        return dataset1
    elif op == int(metricaD.Kp0olinomial):
        dataset1 = pp.RobustScaler().fit_transform(dataset)
        return dataset1


def mstPrim(D, model, closest):
    ##bagEmbed = np.zeros(len(D))
    maxValue = np.max(D)
    Dcopy = D.copy()
    maxDouble = maxValue * 2
    for i in range(len(D)):
        Dcopy[i, i] = maxDouble
    minValue = np.min(Dcopy)
    minCoord = np.unravel_index(np.argmin(Dcopy), np.array(Dcopy).shape)

    ##adiciona conexão no grafo
    print("Gerando Grafo")
    model.g = Graph()
    model.g.add_vertices(len(D))

    ##model.g.vs["label"] = range(0,len(D))

    ##model.g.vs["classes"]= model.Ytrain
    model.g.vs["classes"] = closest

    print("Ajustando paleta rainbow")
    pal = RainbowPalette(n=255)

    ##d=max(model.Ytrain)-min(model.Ytrain)
    d = max(closest) - min(closest)
    matiz = (254 - 31.75) / d

    print("Aplicando Paleta")
    for i in range(len(Dcopy)):
        ##model.g.vs[i]['color'] = pal.get(int(((model.Ytrain[i] - min(model.Ytrain)) * matiz) + 31.75))
        model.g.vs[i]["color"] = pal.get(
            int(((closest[i] - min(closest)) * matiz) + 31.75)
        )

    ##switch
    model.g.add_edge(int(minCoord[0]), int(minCoord[1]))
    model.g.es[model.g.get_eid(int(minCoord[0]), int(minCoord[1]))]["weight"] = minValue

    ##adiciona conexão na bag
    ##bagEmbed[int(minCoord[0])]=1
    ##bagEmbed[int(minCoord[1])]=1
    bagEmbedIndex = []
    bagEmbedIndex.append(int(minCoord[0]))
    bagEmbedIndex.append(int(minCoord[1]))
    Dcopy[int(minCoord[0]), int(minCoord[1])] = maxDouble
    Dcopy[int(minCoord[1]), int(minCoord[0])] = maxDouble

    while len(bagEmbedIndex) != len(D):
        cutDcopy = []
        for i in range(len(bagEmbedIndex)):
            line = np.squeeze(np.asarray(Dcopy[int(bagEmbedIndex[i]), :])).tolist()
            cutDcopy.append(line)
        minValueCut = np.min(cutDcopy)
        minCoordCut = np.unravel_index(np.argmin(cutDcopy), np.array(cutDcopy).shape)
        minCoordi = bagEmbedIndex[int(minCoordCut[0])]
        minCoordj = int(minCoordCut[1])
        if (minCoordi in bagEmbedIndex) and (minCoordj not in bagEmbedIndex):
            ##print("len(bag) = "+str(len(bagEmbedIndex)))
            bagEmbedIndex.append(minCoordj)
            ##adicionar conexão com valor.
            model.g.add_edge(minCoordi, minCoordj)
            model.g.es[model.g.get_eid(minCoordi, minCoordj)]["weight"] = minValueCut
            for i in range(len(bagEmbedIndex)):
                Dcopy[bagEmbedIndex[i], minCoordj] = maxDouble
                Dcopy[minCoordj, bagEmbedIndex[i]] = maxDouble
        ##            for i in range(len(D)):
        ##                Dcopy[minCoordj,bagEmbedIndex[i]]=maxDouble

        Dcopy[minCoordi, minCoordj] = maxDouble
        Dcopy[minCoordj, minCoordi] = maxDouble
        print("len(bag) = " + str(len(bagEmbedIndex)))
        ##print("len(bag) = "+str(len(bagEmbedIndex))+" i = "+str(minCoordi)+" j = "+str(minCoordj))
    print("Plot grafo")
    random.seed(140)
    out = plot(model.g, layout="kk", vertex_size=5)


def mstPrim2(D, model, closest):
    bagEmbed = np.zeros(len(D))
    ##maxValue = np.max(D)
    Dcopy = D.copy()
    ##    maxDouble = maxValue*2
    ##    for i in range(len(D)):
    ##        Dcopy[i,i]=maxDouble
    ##minValue= np.min(Dcopy)
    ##minCoord= np.unravel_index(np.argmin(Dcopy), np.array(Dcopy).shape)

    ##fila = [[] for i in range(len(D))]
    fila = []
    for i in range(1, len(D)):
        for j in range(i):
            tp = [Dcopy[i, j], i, j]
            ##print("i = "+str(i)+" j = "+str(j))
            fila.append(tp)
    fila.sort()

    ##adiciona conexão no grafo
    print("Gerando Grafo")
    model.g = Graph()
    model.g.add_vertices(len(D))

    ##model.g.vs["label"] = range(0,len(D))

    ##model.g.vs["classes"]= model.Ytrain
    model.g.vs["classes"] = closest

    print("Ajustando paleta rainbow")
    pal = RainbowPalette(n=255)

    ##d=max(model.Ytrain)-min(model.Ytrain)
    d = max(closest) - min(closest)
    matiz = (254 - 31.75) / d

    print("Aplicando Paleta")
    for i in range(len(D)):
        ##model.g.vs[i]['color'] = pal.get(int(((model.Ytrain[i] - min(model.Ytrain)) * matiz) + 31.75))
        model.g.vs[i]["color"] = pal.get(
            int(((closest[i] - min(closest)) * matiz) + 31.75)
        )

    tp = fila[0]
    ##adiciona conexão na bag
    bagEmbed[int(tp[1])] = 1
    bagEmbed[int(tp[2])] = 1
    ##bagEmbedIndex=[]
    ##bagEmbedIndex.append(int(tp[1]))
    ##bagEmbedIndex.append(int(tp[2]))

    ##switch
    model.g.add_edge(int(tp[1]), int(tp[2]))
    model.g.es[model.g.get_eid(int(tp[1]), int(tp[2]))]["weight"] = tp[0]

    nt = 1
    soma = sum(bagEmbed)
    while soma != len(D):
        tp = fila[nt]
        minValueCut = tp[0]
        minCoordi = int(tp[1])
        minCoordj = int(tp[2])
        if bagEmbed[minCoordi] != bagEmbed[minCoordj]:
            if bagEmbed[minCoordi] == 1:
                bagEmbed[minCoordj] = 1
            else:
                bagEmbed[minCoordi] = 1
            ##print("len(bag) = "+str(int(sum(bagEmbed)))
            ##adicionar conexão com valor.
            model.g.add_edge(minCoordi, minCoordj)
            model.g.es[model.g.get_eid(minCoordi, minCoordj)]["weight"] = minValueCut
            print("len(bag) = " + str(int(soma)))
            soma = soma + 1
        nt = nt + 1
        ##print("len(bag) = "+str(sum(bagEmbed))+" i = "+str(minCoordi)+" j = "+str(minCoordj))

    print("Plot grafo")
    random.seed(140)
    out = plot(model.g, layout="kk", vertex_size=5)


def knn(D, k=4):

    knn = np.zeros((len(D), k)) - 1

    sortDids = np.argsort(D, axis=1)
    for i in range(0, len(D)):
        for j in range(1, (k + 1)):
            if i != sortDids[i, j]:
                knn[i, (j - 1)] = sortDids[i, j]
    return knn


def verifyX(idKnn, D):
    knnD = idKnn.copy()
    for i in range(len(idKnn)):
        for j in range(10):
            knnD[i, j] = D[int(i), int(idKnn[i, j])]
    return knnD


def verifyY(idKnn, Y):
    knnY = idKnn.copy()
    for i in range(len(idKnn)):
        for j in range(10):
            knnY[i, j] = Y[int(idKnn[i, j])]
    return knnY


def smallworld(knn, p):
    dimL = knnMat.shape[0]
    dimC = knnMat.shape[1]
    for i in range(0, dimL):
        for j in range(1, dimC):
            r = random.random()
            if r < p:
                knn[i, j] = random.randint(0, (dimL - 1))
    return knn


def rbf(euclidean, sigma):
    # euclidean=np.sqrt(np.sum(np.power((x-clusters),2),axis=1))
    return np.exp((-np.power(euclidean, 2)) / (np.power(2 * sigma, 2)))


def buildW(D, knn, sigma=4):
    D = np.asmatrix(D)
    Wi = np.asmatrix(np.zeros(shape=(knn.shape[0], knn.shape[1])))
    for i in range(knn.shape[0]):
        ##print(i)
        for j in range(knn.shape[1]):
            Wi[i, j] = rbf(D[i, int(knn[i, j])], sigma)
    return Wi


def LSR(X, Y, x):
    # revisar resultados erro ~7
    n = len(Y)
    B1 = (np.sum(X * Y) - (n * np.mean(X) * np.mean(Y))) / (
        np.sum(np.power(X, 2)) - n * np.power(np.mean(X), 2)
    )
    B0 = np.mean(Y) - B1 * np.mean(X)
    y = B0 + B1 * x
    return y