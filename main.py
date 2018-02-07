from matplotlib.pyplot import plot
from sklearn import datasets
import csv
import numpy as np

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
    print(dataset)
    dataset=dataset[:,:-1]
    target=dataset[:,-1]
    

