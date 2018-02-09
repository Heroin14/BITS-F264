import matplotlib.pyplot as plt
from sklearn import datasets
import csv
import numpy as np
from sklearn.decomposition import PCA

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

def doPCA(dataset):

    pca=PCA(0.99)
    #components is going to store the number of components in the reduced dimensionatlity data
    dataset=pca.fit_transform(dataset)

    components= dataset[0].size
    print(components)

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

    Plotting(dataset,target)    
    dataset, components =doPCA(dataset)


