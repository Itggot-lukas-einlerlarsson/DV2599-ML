# Kim Lillé and Lukas Einler Larsson
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import time
from datetime import timedelta

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

class spamDetection:
    def __init__(self):
        df = pd.DataFrame()
        trainingSet = pd.DataFrame()
        self.kNNtime = timedelta(0, 0, 0)
        self.SVMtime = timedelta(0, 0, 0)
        self.NaiveBayestime = timedelta(0, 0, 0)
        self.kNNFscore = []
        self.SVMFscore = []
        self.NaiveBayesFscore = []
        return
    def readData(self, data = "spambase/spambase.data"):
        self.df = pd.read_csv(data, header=None, dtype= np.float64)
        return
    def setHeader(self, headerSrc = "spambase/spambase.header"):
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

    def datasetCreation(self):
        """
            Create a dataset similar to sklearn's datasets used on the lab.
        """
        spamDict = {}
        spamDict["frame"] = self.df
        data = []
        target = []
        for i in range(len(self.df)):
            instance = self.df.iloc[i].to_list()
            data.append(instance[:-1])
            target.append(instance[-1])
        spamDict["data"] = np.array(data)
        spamDict["target"] = np.array(target)
        spamDict["target_names"] = np.array(["spam", "ham"])
        spamDict["feature_names"] = np.array(self.df.columns.to_list()[:-1])
        self.spamDict = spamDict

    def run_kNNclassifier(self, n_neighbors, X_train, X_test, y_train, y_test):
        kNNmodel = KNeighborsClassifier(n_neighbors)
        startTime = time.time()
        kNNmodel.fit(X_train, y_train)
        endTime = time.time()
        accuracy = kNNmodel.score(X_test, y_test)
        y_pred = kNNmodel.predict(X_test)
        self.kNNFscore.append(f1_score(y_test, y_pred))
        self.kNNtime += timedelta(seconds=endTime - startTime)
        return accuracy

    def run_SVMclassifier(self, X_train, X_test, y_train, y_test):
        SVMmodel = SVC()
        startTime = time.time()
        SVMmodel.fit(X_train, y_train)
        endTime = time.time()
        accuracy = SVMmodel.score(X_test, y_test)
        y_pred = SVMmodel.predict(X_test)
        self.SVMFscore.append(f1_score(y_test, y_pred))
        self.SVMtime += timedelta(seconds=endTime - startTime)
        return accuracy


    def run_NaiveBayesClassifier(self, X_train, X_test, y_train, y_test):
        NBmodel = ComplementNB()
        startTime = time.time()
        NBmodel.fit(X_train, y_train)
        endTime = time.time()
        accuracy = NBmodel.score(X_test, y_test)
        y_pred = NBmodel.predict(X_test)
        self.NaiveBayesFscore.append(f1_score(y_test, y_pred))
        self.NaiveBayestime += timedelta(seconds=endTime - startTime)
        return accuracy


    def run_stratifiedKfoldTest(self, n = 10):
        """ page 349, 350
            runs stratified k fold test
            During each learning iteration, for each algorithm: accuracy, fscore and time is measured
        """
        X, y = self.spamDict["data"], self.spamDict["target"]
        skf = StratifiedKFold(n_splits=n, random_state=None)
        accuracy_kNN = []
        accuracy_SVM = []
        accuracy_NaiveBayes = []
        foldnr = 1
        for train_index, test_index in skf.split(X,y): # X is the feature set and y is the target
            print("training... foldnr:", foldnr)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            accuracy_kNN.append(self.run_kNNclassifier(20, X_train, X_test, y_train, y_test))
            accuracy_SVM.append(self.run_SVMclassifier(X_train, X_test, y_train, y_test))
            accuracy_NaiveBayes.append(self.run_NaiveBayesClassifier(X_train, X_test, y_train, y_test))
            foldnr += 1

        # run test to see if there is difference between the algorithms performance
        ranks = self.get_friedmanRanks(accuracy_kNN, accuracy_SVM, accuracy_NaiveBayes)
        self.printTable(np.array(accuracy_kNN), np.array(accuracy_SVM), np.array(accuracy_NaiveBayes), ranks)
        mean_TOTrank, sum_squaredDiff, sum_squaredDiff_nk, mean_ranks = self.run_FriedmanTest(ranks)
        CV = self.run_calcCriticalValue(k = 3, n = 10, siglevel = 0.05)

        #see if significant difference
        if sum_squaredDiff > CV:
            print(f"Friedman Test: There is a significant difference between the algorithms, {sum_squaredDiff} > {CV}")
            print("Running running Nemeyi test:", end = " ")
            self.run_nemenyiTest(mean_ranks)
        else:
            print("Friedman Test: There is no significant difference between the algorithms")


    def get_friedmanRanks(self, accuracy_kNN, accuracy_SVM, accuracy_NaiveBayes, k = 3):
        """ Page 355, 356 in the course book
            idea: rank the performance of all k algorithms per data set
            from best performance (rank 1) to worst (rank k)

            course books symbols:
            R_i_j = rank of the j:th algorithm on the i:th dataset
            R_j = average rank of the j:th algorithm
            Null hypothesis, H_0: all algorithms perform the same
        """
        ranks_kNN = []
        ranks_SVM = []
        ranks_NaiveBayes = []
        for i in range(len(accuracy_kNN)):
            accuracies = []
            accuracies.append((accuracy_kNN[i], "kNN"))
            accuracies.append((accuracy_SVM[i], "SVM"))
            accuracies.append((accuracy_NaiveBayes[i], "NaiveBayes"))
            accuracies = sorted(accuracies, reverse=True)
            rank = 1
            for value in accuracies:
                if value[1] == "kNN":
                    ranks_kNN.append(rank)
                if value[1] == "SVM":
                    ranks_SVM.append(rank)
                if value[1] == "NaiveBayes":
                    ranks_NaiveBayes.append(rank)
                rank += 1
        ranks = {}
        ranks["kNN"] = np.array(ranks_kNN)
        ranks["SVM"] = np.array(ranks_SVM)
        ranks["NaiveBayes"] = np.array(ranks_NaiveBayes)
        return ranks

    def run_FriedmanTest(self, ranks, k = 3):
        """ page 355
        This Function calculates the three points from the course book:
        1) The average rank
        2) The sum of square differences, measuring the spread of rank centroids
        3) The sum of square differences, measuring the spread of all rank
        """
        print()
        print("-" * 60)
        print("\t\tkNN\t\tSVM\t\tNaive Bayes")
        print("-" * 60)
        # Friedman: Average rank
        mean_kNNrank = ranks["kNN"].mean() # 2.1
        mean_SVMrank = ranks["SVM"].mean() # 1.6
        mean_NaiveBayesrank = ranks["NaiveBayes"].mean() # 2.3
        mean_ranks = [mean_kNNrank, mean_SVMrank, mean_NaiveBayesrank]
        print(f"avg rank\t{mean_kNNrank}\t\t{mean_SVMrank}\t\t{mean_NaiveBayesrank}")
        mean_TOTrank = (k+1)/2 # 1)

        # sum of squared differences # 2)
        sum_squaredDiff = (mean_kNNrank - mean_TOTrank)**2
        sum_squaredDiff += (mean_SVMrank - mean_TOTrank)**2
        sum_squaredDiff += (mean_NaiveBayesrank - mean_TOTrank)**2
        sum_squaredDiff *= len(ranks["kNN"])

        sum_squaredDiff_nk = 0 # 3)
        for i in range(len(ranks["kNN"])):
            sum_squaredDiff_nk += (ranks["kNN"][i] - mean_TOTrank)**2
            sum_squaredDiff_nk += (ranks["SVM"][i] - mean_TOTrank)**2
            sum_squaredDiff_nk += (ranks["NaiveBayes"][i] - mean_TOTrank)**2
        sum_squaredDiff_nk *= 1/(len(ranks["kNN"])*(len(ranks)-1))

        print("-" * 60)
        print("avg total rank:", mean_TOTrank)
        print("the sum of squared differences(spread of rank centroids):", sum_squaredDiff)
        print("the sum of squared differences nk(spread over all ranks):", sum_squaredDiff_nk)
        return (mean_TOTrank, sum_squaredDiff, sum_squaredDiff_nk, mean_ranks)

    def run_calcCriticalValue(self, k, n, siglevel):
        """ page 356
            ref: P. Flach, “12 Machine Learning experiments” in Machine Learning: The Art and Science of Algorithms that Make Sense of Data,
            1st ed, Cambridge, United Kingdom, Cambridge university press, 2012
            returns the critical value for k = 3, n = 10
        """
        return 7.8

    def run_nemenyiTest(self, mean_ranks, k = 3, n = 10, sigLevel = 0.05):
        """ page 356
            p_sigLevel = 3.314/math.sqrt(2)
            value 3.314 taken from https://www.real-statistics.com/statistics-tables/studentized-range-q-table/
            alpha level 0.05
        """
        p_sigLevel = 2.343
        CD = p_sigLevel * math.sqrt( (k*(k+1))/(6*n) )
        mean_ranks = sorted(mean_ranks)
        criticalDifferenceAlgorithms = [] # holds all algorithm's that have a critical difference
        if max(mean_ranks) - min(mean_ranks) > CD:
            str(criticalDifferenceAlgorithms.append(max(mean_ranks)))
            str(criticalDifferenceAlgorithms.append(min(mean_ranks)))
            if mean_ranks[1] - mean_ranks[0] > CD:
                str(criticalDifferenceAlgorithms.append(mean_ranks[1]))
            if mean_ranks[1] not in criticalDifferenceAlgorithms:
                if mean_ranks[2] - mean_ranks[1] > CD:
                    str(criticalDifferenceAlgorithms.append(mean_ranks[1]))
            print(f"There is a significant difference between the algorithms with average ranks: ",criticalDifferenceAlgorithms)
        else:
            print(f"There is no significant difference between the algorithms")



    def printTable(self, accuracy_kNN, accuracy_SVM, accuracy_NaiveBayes, ranks = None):
        """ Page 350.
            printing a table similar to the figure 12.4( and 12.8) in the course book
        """
        print("-" * 60)
        print("Fold\t\tkNN\t\tSVM\t\tNaive Bayes")
        print("-" * 60)
        np.set_printoptions(precision=5)
        self.kNNFscore = np.array(self.kNNFscore)
        self.SVMFscore = np.array(self.SVMFscore)
        self.NaiveBayesFscore = np.array(self.NaiveBayesFscore)
        for i in range(len(accuracy_kNN)):
            print(i+1, end = " Accuracy:\t")
            print(np.round(accuracy_kNN[i], 5), end = "\t\t")
            print(np.round(accuracy_SVM[i], 5),end = "\t\t")
            print(np.round(accuracy_NaiveBayes[i], 5))
            print(end = "  F-score:\t")
            print(np.round(self.kNNFscore[i], 5), end = "\t\t")
            print(np.round(self.SVMFscore[i], 5), end = "\t\t")
            print(np.round(self.NaiveBayesFscore[i], 5))
            print(end = "  ranks:\t")
            print(ranks["kNN"][i], end = "\t\t")
            print(ranks["SVM"][i],end = "\t\t")
            print(ranks["NaiveBayes"][i])



        print("-" * 60)
        print(f"avg accuracy\t{np.round(accuracy_kNN.mean(), 5)}\t\t{np.round(accuracy_SVM.mean(), 5)}\t\t{np.round(accuracy_NaiveBayes.mean(), 5)}")
        print(f"stdev\t\t{np.round(accuracy_kNN.std(), 5)}\t\t{np.round(accuracy_SVM.std(), 5)}\t\t{np.round(accuracy_NaiveBayes.std(), 5)}")
        print(f"learning time\t{self.kNNtime}\t{self.SVMtime}\t{self.NaiveBayestime}")



def main():
    spamDe = spamDetection()
    spamDe.readData()
    spamDe.setHeader()
    spamDe.clean()
    spamDe.datasetCreation()
    spamDe.run_stratifiedKfoldTest()


if __name__ == '__main__':
    main()
