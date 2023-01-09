# Kim Lillé and Lukas Einler Larsson
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
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
            qcut and cut into intervals and then into ints didnt help with seg fault on big data
        """
        pass
        # discretize
        # for column in self.downsampledData.columns:
        #     if column == "Class":
        #         continue
        #     # self.downsampledData.sort_values(column)
        #     # self.downsampledData[column] = pd.qcut(self.downsampledData[column], q = 20).cat.codes
        #     self.downsampledData[column] = pd.cut(self.downsampledData[column], bins = 20).cat.codes
        # self.downsampledData.astype('int32')

        # standardization
        # rs = preprocessing.RobustScaler()
        # tempArray = rs.fit_transform(self.downsampledData) # returns numpy array
        # self.downsampledData = pd.DataFrame(tempArray, index = self.downsampledData.index, columns = self.downsampledData.columns)

    def testClusterModel(self, distance, minSampleSize, data):
        """
            Test: ändra parametrar o försöka få alla fraud instanser till samma outlier kluster och
            samtidigt få alla benign transactions till andra kluster.

            DBSCAN desc: https://towardsdatascience.com/how-dbscan-works-and-why-should-i-use-it-443b4a191c80
        """
        dbscanModel = DBSCAN(eps = distance, min_samples=minSampleSize)
        TN = 0
        FN = 0
        TP = 0
        FP = 0

        if "Class" in data.columns: # if running with downsampledData and all fraud instances
            dbscanModel.fit(data.loc[:, data.columns != "Class"])
            TP, FP, TN, FN = self.test(dbscanModel.labels_, data)
        else: # if running with cleaneDf[:length]
            dbscanModel.fit(data)
            TP, FP, TN, FN = self.test(dbscanModel.labels_, self.df)

        return TP, FP, TN, FN

    def test(self, labels, data):
        """
            Test accuracy of clustering model

            Labels DBscan: -1 = outlier, all other numbers belongs to cluster
        """
        TN = 0
        FN = 0
        TP = 0
        FP = 0
        for i, v in enumerate(labels):
            instance = data.iloc[i]
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
        return TP, FP, TN, FN

    def printStatistics(self, TP, FP, TN, FN):
        print("Positive indicates Fraud, negative indicates not Fraud")
        print("-"*20)
        print("True Positive:", TP)
        print("False Positive:", FP)
        print("True Negative:", TN)
        print("False Negative:", FN)
        print("Total Amount:", TP+FP+TN+FN)
        print("-"*20)
        print("Accuracy:", (TP + TN) / (TP + FP + FN + TN))
        print("Precision:", TP / (TP + FP))
        print("Recall:", TP / (TP + FN))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        print("F1-Score: ", 2 * (precision * recall)/(precision + recall))

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

    def elbowMethod(self):
        # https://towardsdatascience.com/explaining-dbscan-clustering-18eaf5c83b31
        nbrs = NearestNeighbors(n_neighbors=2).fit(self.df)
        neigh_dist, neigh_ind = nbrs.kneighbors(self.df)
        sort_neigh_dist = np.sort(neigh_dist, axis=0)

        k_dist = sort_neigh_dist[:]
        plt.plot(k_dist)
        plt.axhline(y=150, linewidth=1, linestyle='dashed', color='k')
        plt.axhline(y=350, linewidth=1, linestyle='dashed', color='k')
        plt.ylabel("k-NN distance")
        plt.xlabel("Sorted observations (4th NN)")
        plt.show()

def test_SampleSize():
    """Test for best distance for the given sampleSizes that could be read using the elbow method."""
    fraudDe = fraudDetection()
    fraudDe.readData()
    fraudDe.setHeader()
    for j in [150, 200, 250, 300, 350]:
        for i in range(2, 12):
            print(f"i = {i}, j = {j}")
            TP = []
            FP = []
            TN = []
            FN = []
            for y in range(20):
                fraudDe.downSample(length = 20000)
                tp, fp, tn, fn = fraudDe.testClusterModel(distance = i, minSampleSize = j, data = fraudDe.downsampledData)
                TP.append(tp)
                FP.append(fp)
                TN.append(tn)
                FN.append(fn)
            with open(f"test_SampleSize_{j}.txt", "a") as f:
                f.write("-"*30 + f"\nDistance: {str(i)}\nSample Size: {j} \n\nTP:\n")
                for tp in TP:
                    f.write(str(tp) + "\n")
                f.write("FP:\n")
                for fp in FP:
                    f.write(str(fp) + "\n")
                f.write("TN:\n")
                for tn in TN:
                    f.write(str(tn) + "\n")
                f.write("FN:\n")
                for fn in FN:
                    f.write(str(fn) + "\n")

def main():
    fraudDe = fraudDetection()
    fraudDe.readData()
    fraudDe.setHeader()
    fraudDe.clean()
    fraudDe.downSample(length = 20000)
    fraudDe.preprocess()
    tp, fp, tn, fn = fraudDe.testClusterModel(distance = 7, minSampleSize = 200, data = fraudDe.downsampledData)
    fraudDe.printStatistics(tp, fp, tn, fn)
    fraudDe.printConfusionMatrix(tp, fp, tn, fn)

main()
