# Kim Lillé and Lukas Einler Larsson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
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
        # self.df = self.df.drop("Class")
        print(self.df.columns)

    def testKMeans(self):
        kMeans = KMeans(n_clusters=3, random_state=0)
        kMeans.fit(self.df["v1"].to_numpy())
        pca = PCA(n_components=2)
        pca.fit(self.df["v1"].to_numpy())
        X_reduced = pca.transform(self.df.to_numpy())
        plt.scatter(X_reduced[:, 0],
            X_reduced[:, 1],
            c=kMeans.labels_,
            cmap=plt.cm.get_cmap('spring', 3),
            alpha=0.5);
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show();



def main():
    fraudDe = fraudDetection()
    fraudDe.readData()
    fraudDe.setHeader()
    fraudDe.clean()
    fraudDe.testKMeans()



main()
