import matplotlib.pyplot as plt
from sklearn import datasets
import csv
import numpy as np
from sklearn.decomposition import PCA

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

    #plt.plot(dataset,target)
    dataset, components =doPCA(dataset)


