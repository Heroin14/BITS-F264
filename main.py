import matplotlib.pyplot as plt
from sklearn import datasets
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

def Plotting(dataset, target):

    pca=PCA(n_components=2)
    dataTransformed= pca.fit_transform(dataset)

    #dataTransformed as two components

    colors=[]
    for i in target:
        if(i==0):
            colors.append('r')
        else:
            colors.append('b')
    colors=np.array(colors)

    plt.xlabel="Component1"
    plt.ylabel="Component2"
    plt.title="Scatter Plot"
    plt.scatter(dataTransformed[:,0],dataTransformed[:,1],s=5,c=colors)
    plt.legend()
    plt.show()

"""def openFile():
    f=open("Accuracies.txt","w+")
    return f"""
"""def closeFile(f):
    f.close()"""

def DecisionTree(dataset , target):
    
    clf=DecisionTreeClassifier(random_state=42)
    sum=0

    for i in range(10):
        X_train, X_test,y_train, y_test=train_test_split(dataset, target, test_size=0.1)
        clf.fit(X_train,y_train)
        sum+= clf.score(X_test,y_test)

    avg=sum/10
    f=openFile()
    f.write("Decision tree classifier " )
    f.write(str(avg))
    f.write("\n")
    closeFile(f)



def doPCA(dataset):

    pca=PCA(0.99)
    #components is going to store the number of components in the reduced dimensionatlity data
    dataset=pca.fit_transform(dataset)
    components= dataset[0].size

    #altered database
    return dataset, components

if __name__=='__main__':
    
    dataset=[]            
    with open('dataA2.csv','r+') as inf:
        reader= csv.reader(inf, delimiter=',')
        for row in reader:
            row_wise=[]
            for ele in row:
                row_wise.append(int(ele))
        
            dataset.append(row_wise)

    dataset=np.array(dataset)
    target=dataset[:,-1]
    dataset=dataset[:,:-1]

    #Plotting(dataset,target)
    #Plot is done    
    dataset, components =doPCA(dataset)
    DecisionTree(dataset,target)

