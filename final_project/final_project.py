# Kim Lillé and Lukas Einler Larsson
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

    def testClusterModels(self, n = 7):
        """
            Testar runt med clustering algorithmer, Just nu clustrar den alla instanser, tar fram index för varje fraud instans,
            och printar ut vilket kluster den instansen fick

            todo: första parametrar o testa runt o försöka få alla fraud instanser till samma kluster.
        """
        # kMeans = KMedoids(n_clusters=10, random_state=0)
        kMeans = KMeans(n_clusters= n, n_init=10) # går snabbt
        # kMeans = DBSCAN(eps=0.5, min_samples=5) # långsam men funka tog ca 5 min
        # kMeans = SpectralClustering(n_clusters=n, assign_labels='discretize', random_state=0) # Unable to allocate 600. GiB for an array with shape (283726, 283726) and data type float64
        # kMeans = AgglomerativeClustering() #Unable to allocate 300. GiB for an array with shape (40250079675,) and data type float64
        kMeans.fit(self.cleaneDf.to_numpy())
        fraud_labels = []
        fraudIndices = self.df[self.df["Class"] == 1.0].index.to_numpy()
        for index in fraudIndices:
            fraud_labels.append(kMeans.labels_[index])
        print(fraud_labels)

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
        plt.show();



def main():
    fraudDe = fraudDetection()
    fraudDe.readData()
    fraudDe.setHeader()
    fraudDe.clean()
    fraudDe.testClusterModels()
    # fraudDe.plotFraudData()



main()
