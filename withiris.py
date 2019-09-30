import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#minmaxscaler is suitable for data which is not normally distributed
from scipy.stats import normaltest
from sklearn.decomposition import PCA #for MDS
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
features = ['sepal length','sepal width','petal length','petal width','target']
df = pd.read_csv(url,names = features)
features.pop()#remove target colnumn not actually part of features
x = df[features].values #returns a straight up np array of feature values
y = df['target'] #target labels
#print(df.tail())
#print(type(x))

print('checking for normal distribution')
#check for normal distribution to determine which features are normalized and which ones 
#are not
z,pval = normaltest(np.array(x)) 
i = -1
for p in pval: #iterat through pvals of features
    i = i+1
    if p<0.05:
        print(features[i],'is not normally distributed')
    else:
        print(features[i],'is normally distributed')

print('\nnormalizing features')
norm_x1 = MinMaxScaler().fit_transform(x) #with minmaxscaler
norm_x2 = StandardScaler().fit_transform(x) #normalize with standard scaler

data_norm = norm_x1 #select this data
#print(norm_x)
def in_2D():
    n = 2 #number of dimensions to features to
    print('scaling features to',n,'dimensions')
    pca = PCA(n_components=n) # scale normalized data to n dimensions
    p_comp = pca.fit_transform(data_norm)
    #make p_comp values a dataframe from np type
    pdf_comp = pd.DataFrame(data = p_comp,columns = ['pc1','pc2'])
    pdf_comp= pd.concat([pdf_comp,y],axis = 1)#concatenate pcomps and target labels
    #print(pdf_comp)

    #filter out and categorize the pca values (components) according to their classe or target labels
    class_setosa = (pdf_comp[pdf_comp['target'].str.match('Iris-setosa')])
    class_virginica = (pdf_comp[pdf_comp['target'].str.match('Iris-virginica')])
    class_versicolor = (pdf_comp[pdf_comp['target'].str.match('Iris-versicolor')])

    #now separate into their xs and ys
    #print(class_setosa)
    print('explained variance',pca.explained_variance_ratio_)

    #visualize data
    print('visualizing data in 2D')
    fig= plt.figure(facecolor = 'black')
    ax = fig.add_subplot(1,1,1,frameon = False)
    ax.scatter(class_setosa['pc1'].values,class_setosa['pc2'].values,c ='red')
    ax.scatter(class_virginica['pc1'].values,class_virginica['pc2'].values,c ='blue')
    ax.scatter(class_versicolor['pc1'].values,class_versicolor['pc2'].values,c ='green')
    ax.legend(['iris-setosa','iris-virginica','iris-versicolor'])
    ax.grid()
    #plt.plot(p_comp)
    plt.show()

def in_3D():
    print('visualizing in 3d')
    pca = PCA(n_components = 3)
    p_comp = pca.fit_transform(data_norm)
    
    print('explained variance',pca.explained_variance_ratio_)
    pdf_comp = pd.DataFrame(data = p_comp,columns = ['pc1','pc2','pc3']) #create dataframe
    pdf_comp = pd.concat([pdf_comp,y],axis =1)#concatenate pcaomps with target labels

    #filter out and categorize the pca values (components) according to their classe or target labels
    class_setosa = (pdf_comp[pdf_comp['target'].str.match('Iris-setosa')])
    class_virginica = (pdf_comp[pdf_comp['target'].str.match('Iris-virginica')])
    class_versicolor = (pdf_comp[pdf_comp['target'].str.match('Iris-versicolor')])

    #print(class_versicolor)
    fig= plt.figure(facecolor = 'black')
    ax = fig.add_subplot(1,1,1,projection = '3d',facecolor = 'black')
    ax.scatter(class_setosa['pc1'].values,class_setosa['pc2'].values,class_setosa['pc3'].values,c ='red')
    ax.scatter(class_virginica['pc1'].values,class_virginica['pc2'].values,class_virginica['pc3'].values,c ='blue')
    ax.scatter(class_versicolor['pc1'].values,class_versicolor['pc2'].values,class_versicolor['pc3'].values,c ='green')
    ax.legend(['iris-setosa','iris-virginica','iris-versicolor'])
    ax.grid()
    #plt.plot(p_comp)
    plt.show()

if __name__ == '__main__':
    in_2D()
    in_3D()


























