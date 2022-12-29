# Kim Lillé and Lukas Einler Larsson
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.utils import resample
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

    def plotKDistanceGraph(self, X, k):
        """
            https://stackoverflow.com/questions/43160240/how-to-plot-a-k-distance-graph-in-python
        """
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
        plt.hist(kn_distance, bins = 30)
        plt.ylabel('n')
        plt.xlabel('Epsilon distance')
        plt.show()

    def downSample(self, length = 10000):
        """
            Randomly extract instances that are benign
            then concat the fraud instances to the dataframe
        """
        self.downsampledData = resample(self.df[self.df["Class"] == 0.0],
                     replace=True,
                     n_samples=length)
                     # random_state=420) #if same random set is wanted over multiple runs
        self.downsampledData = pd.concat([self.downsampledData, self.df[self.df["Class"] == 1.0]],axis=0)
        # self.downsampledData = self.downsampledData.drop(["Class"], axis = 1)
        self.downsampledData = self.downsampledData.drop(["Time"], axis = 1)
        self.downsampledData = self.downsampledData.drop(["Amount"], axis = 1)

    def preprocess(self):
        """
            https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
            Robust scaler is not sensitive to outliers but not better anyway
        """
        pass
        # rs = preprocessing.RobustScaler()
        # tempArray = rs.fit_transform(self.cleaneDf) # returns numpy array
        # self.cleaneDf = pd.DataFrame(tempArray, index = self.cleaneDf.index, columns = self.cleaneDf.columns)

    def testClusterModel(self, distance, minSampleSize, data):
        """
            Test: ändra parametrar o försöka få alla fraud instanser till samma outlier kluster och
            samtidigt få alla benign transactions till andra kluster.

            DBSCAN desc: https://towardsdatascience.com/how-dbscan-works-and-why-should-i-use-it-443b4a191c80
        """
        dbscanModel = DBSCAN(eps = distance, min_samples=minSampleSize)
        if "Class" in data.columns: # if running with downsampledData and all fraud instances
            dbscanModel.fit(data.loc[:, data.columns != "Class"])
            self.test(dbscanModel.labels_, data)
        else: # if running with cleaneDf[:length]
            dbscanModel.fit(data)
            self.test(dbscanModel.labels_, self.df)

    def test(self, labels, data):
        """
            Test accuracy of clustering model

            Labels DBscan: -1 = outlier, all other numbers belongs to cluster
        """
        print("-"*20)
        TN = 0
        FN = 0
        TP = 0
        FP = 0
        for i, v in enumerate(labels):
            instance = data.iloc[i]
            # print(instance)
            # break
            if v != -1: # if instance is not outlier and fraud
                if instance[-1] == 0.0: #if not actual fraud
                    TN += 1
                else:
                    FN += 1
            else: # if outlier
                if instance[-1] == 1.0: #if actual fraud
                    TP += 1
                else:
                    FP += 1
        print("Positive indicates Fraud, negative indicates not Fraud")
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
    fraudDe.downSample(length = 20000)
    fraudDe.preprocess()
    fraudDe.testClusterModel(distance = 5, minSampleSize = 10, data = fraudDe.downsampledData)
    # fraudDe.testClusterModel(distance = 5, minSampleSize = 10, data = fraudDe.cleaneDf[:20000])
    # fraudDe.plotKDistanceGraph(fraudDe.cleaneDf[:1000].to_numpy(), 20)

main()
