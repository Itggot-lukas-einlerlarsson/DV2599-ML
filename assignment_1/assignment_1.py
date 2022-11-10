#https://github.com/sampepose/SpamClassifier
#https://rstudio-pubs-static.s3.amazonaws.com/89589_4d2a66ce5276434c9f56a54df5739e85.html
#https://www2.stat.duke.edu/courses/Spring06/sta293.3/topic9/spam_name.txt
#https://en.wikipedia.org/wiki/Receiver_operating_characteristic
#https://en.wikipedia.org/wiki/Confusion_matrix

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer


class spamDetection:
    def __init__(self):
        df = pd.DataFrame()
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

    def plotData(self):
        """
            plots data
        """
        fig=plt.figure()
        ax=fig.add_axes([0,0,1,1])
        self.avg_spam_df = self.df[:1679].mean() #:1679 is spam
        self.avg_ham_df = self.df[1679:].mean() #1679: is ham
        ax.scatter(self.df.columns[:-4], self.avg_spam_df.to_list()[:-4], color='r')
        ax.scatter(self.df.columns[:-4], self.avg_ham_df.to_list()[:-4], color='b')
        ax.set_xlabel('column')
        ax.set_ylabel('value')
        ax.set_title('scatter plot')
        # avg_spam_group = 0
        # for value in avg_spam_df.to_list()[:24]:
        #     avg_spam_group += value
        # print(avg_spam_group/len(avg_spam_df.to_list()[:24]))
        # avg_ham_group = 0
        # for value in avg_ham_df.to_list()[24:-4]:
        #     avg_ham_group += value
        # print(avg_ham_group/len(avg_ham_df.to_list()[24:-4]))
        plt.show()

    def transformData(self):
        """
            Transforms the data
        """
        length = len(self.df)
        # intervalIndex = pd.interval_range(start=0, freq=5, end=20, closed='left')
        for column in self.df.columns:
            self.df = self.df.sort_values(column)
            # self.df[column] = pd.cut(self.df[column], bins = 10, precision = 1)
            self.df[column] = pd.cut(self.df[column], bins = 5, precision = 1)
        print(self.df)




    def LGG_Set(self, D):
        """ s. 108 in the course book
            Input: Data D,
            Output Logical expression H
        """
        x = D.iloc[0].to_list()
        H = x
        for i in range(1, 1679): #len(D)):
            x = D.iloc[i].to_list()
            H = self.LGG_Conj(H, x)
        return H

    def LGG_Conj(self, H, x):
        """ s. 110 in the course book
            input: conjunctions H, x
            output: conjunction H
        """
        for i in range(len(x)):
            if H[i] == "?": # feature already considered general
                continue
            if x[i] != H[i]: # no conjunction
                H[i] = "?" # feature is general
        return H

    def test(self):
        H = self.LGG_Set(self.df) # get hypothesis
        print("H:", H)
        indices = []
        for i,value in enumerate(H):
            if value != "?": # feature is general and not interesting
                indices.append(i)
        print(indices)
        total_amount = 0
        ham_detected = 0
        for i in range(0,len(self.df)):
            spam = True
            lst = self.df.iloc[i].to_list()
            for j in indices:
                if H[j] != lst[j]:
                    spam = False
            if spam == False:
                ham_detected += 1
            total_amount += 1
        print("ham_detected:", ham_detected)
        print("total_amount:", total_amount)

def main():
    spamDe = spamDetection()
    spamDe.readData()
    spamDe.setHeader()
    spamDe.clean()

    # check how many instances is classed as spam
    # count = 0
    # for i in range(len(spamDe.df)):
    #     if spamDe.df.iloc[i][-1] == 1: # if classed as spam
    #         count += 1
    # print("amount of spam:", count)
    spamDe.transformData()
    # spamDe.plotData()
    # print(spamDe.LGG_Set(spamDe.df))
    spamDe.test()

if __name__ == '__main__':
    main()
