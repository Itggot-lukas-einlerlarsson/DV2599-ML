# Kim Lillé and Lukas Einler Larsson
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
sns.set_theme()


class fraudDetection:
    def __init__(self):
        df = pd.DataFrame()
        return
    def readData(self, data = "dataset/creditcard.csv"):
        self.df = pd.read_csv(data, header=None, dtype= np.float64)
        return
    def setHeader(self, headerSrc = "dataset/creditcard.header"):
        header = ""
        with open(headerSrc) as f:
            header = f.readlines()
        for i, _ in enumerate(header):
            header[i] = header[i].strip()
            header[i] = header[i].strip(',')
        self.df.columns = header
        return
    def clean(self):
        self.df.dropna()
        self.df = self.df.drop_duplicates()
        self.cleaneDf = self.df.drop(["Class"], axis = 1)
        self.cleaneDf = self.cleaneDf.drop(["Time"], axis = 1)
        self.cleaneDf = self.cleaneDf.drop(["Amount"], axis = 1)
        print(self.cleaneDf.columns)

    def preprocess(self):
        tf = Normalizer()
        self.cleaneDf = self.cleaneDf.round(3)
        tf.transform(self.cleaneDf)
        print(self.cleaneDf)

    def testClusterModels(self, n = 7):
        """
            Testar runt med clustering algorithmer, Just nu clustrar den alla instanser, tar fram index för varje fraud instans,
            och printar ut vilket kluster den instansen fick

            todo: första parametrar o testa runt o försöka få alla fraud instanser till samma kluster.

            DBSCAN desc: https://towardsdatascience.com/how-dbscan-works-and-why-should-i-use-it-443b4a191c80
        """
        # kMeans = KMedoids(n_clusters=10, random_state=0)
        # kMeans = KMeans(n_clusters= n, n_init=10) # går snabbt
        kMeans = DBSCAN(eps=0.69, min_samples=5) # långsam men funka tog ca 5 min
        # kMeans = SpectralClustering(n_clusters=n, assign_labels='discretize', random_state=0) # Unable to allocate 600. GiB for an array with shape (283726, 283726) and data type float64
        # kMeans = AgglomerativeClustering() #Unable to allocate 300. GiB for an array with shape (40250079675,) and data type float64
        kMeans.fit(self.cleaneDf.to_numpy())
        fraud_labels = []
        fraudIndices = self.df[self.df["Class"] == 1.0].index.to_numpy()
        for index in fraudIndices:
            fraud_labels.append(kMeans.labels_[index])
        print(fraud_labels)
        self.test(kMeans.labels_)

    def plotKDistanceGraph(self, X, k):
        """
            https://stackoverflow.com/questions/43160240/how-to-plot-a-k-distance-graph-in-python
        """
        # X = self.cleaneDf.to_numpy()  #.to_numpy()
        kn_distance = []
        for i in range(len(X)):
            eucl_dist = []
            for j in range(len(X)):
                eucl_dist.append(
                    math.sqrt(
                        ((X[i,0] - X[j,0]) ** 2) +
                        ((X[i,1] - X[j,1]) ** 2)))

            eucl_dist.sort()
            kn_distance.append(eucl_dist[k])
        plt.hist(kn_distance, bins = 50)
        plt.ylabel('n')
        plt.xlabel('Epsilon distance')
        plt.show()


    def plotFraudData(self, n = 7):
        kMeans = KMeans(n_clusters= n, n_init=10)
        kMeans.fit(self.df[self.df["Class"] == 1.0].to_numpy())
        pca = PCA(n_components=2)
        pca.fit(self.df[self.df["Class"] == 1.0].to_numpy())
        X_reduced = pca.transform(self.df[self.df["Class"] == 1.0].to_numpy())
        plt.scatter(X_reduced[:, 0],
            X_reduced[:, 1],
            c=kMeans.labels_,
            cmap=plt.cm.get_cmap('spring', n),
            alpha=0.5);
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()

    def test(self, labels):
        """
            test accuracy of clustering model
        """
        print("-"*20)
        TN = 0
        FN = 0
        TP = 0
        FP = 0
        for i in range(len(self.df)):
            lst = self.df.iloc[i].to_list()
            if labels[i] != -1: # if instance is not outlier and fraud
                if lst[-1] == 0.0: #if not actual fraud
                    TN += 1
                else:
                    FN += 1
            else: # if outlier
                if lst[-1] == 1.0: #if actual fraud
                    TP += 1
                else:
                    FP += 1
        print("Positive indicates Spam, negative indicates Ham")
        print("-"*20)
        print("True Positive:", TP)
        print("False Positive:", FP)
        print("True Negative:", TN)
        print("False Negative:", FN)
        print("Total Amount:", TP+FP+TN+FN)
        print("-"*20)
        print("Precision:", TP / (TP + FP))
        print("Recall:", TP / (TP + FN))
        print("Accuracy:", (TP + TN) / (TP + FP + FN + TN))
        self.printConfusionMatrix(TP, FP, TN, FN)

    def printConfusionMatrix(self, TP, FP, TN, FN):
        """ page 54 in the course book
        """
        print("-"*60)
        print("\t\tPredicted +\tPredicted -")
        print("-"*60)
        print(f"Actual +\t\t{TP}\t\t{FN}\t\t{TP+FN}")
        print(f"Actual -\t\t{FP}\t\t{TN}\t\t{FP+TN}")
        print(f"\t\t\t{TP+FP}\t\t{FN+TN}\t\t{TP+TN+FN+FP}")
        print("-"*60)


def main():
    fraudDe = fraudDetection()
    fraudDe.readData()
    fraudDe.setHeader()
    fraudDe.clean()
    fraudDe.preprocess()
    fraudDe.testClusterModels()
    # fraudDe.plotKDistanceGraph(fraudDe.cleaneDf[1:2000].to_numpy(), 5)
    # fraudDe.plotFraudData()



main()
