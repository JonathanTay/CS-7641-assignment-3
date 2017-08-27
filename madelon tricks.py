# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 13:08:50 2017

@author: JTay
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from helpers import  nn_arch,nn_reg,ImportanceSelect
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
out = './PCA/'
cmap = cm.get_cmap('Spectral') 

np.random.seed(0)


madelon = pd.read_hdf('./BASE/datasets.hdf','madelon')        
madelonX = madelon.drop('Class',1).copy().values
madelonY = madelon['Class'].copy().values
scaler =StandardScaler()

madelon_test = pd.read_hdf('./BASE/datasets.hdf','madelon')        
madelon_tstX = madelon_test.drop('Class',1).copy().values
madelon_tstY = madelon_test['Class'].copy().values
from sklearn.ensemble import RandomForestClassifier



madelonX = scaler.fit_transform(madelonX)
madelon_tstX = scaler.transform(madelon_tstX)


#Reproduce best estimator so far
#if __name__=='__main__':
#    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
#    filtr = ImportanceSelect(rfc)
#    grid ={'filter__n':[20],'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
#    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
#    pipe = Pipeline([('filter',filtr),('NN',mlp)])
#    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)    
#    gs.fit(madelonX,madelonY)
#    print('Best CV Score {}'.format(gs.best_score_))
#    print('Test Score {}'.format(gs.score(madelon_tstX,madelon_tstY)))
#    rf_features = gs.best_estimator_.steps[0][1].model.feature_importances_.argsort()[::-1][:20]
    
    
# Use PCA to find true correct featuers
pca = PCA(random_state=5,n_components=500)
pca.fit(madelonX)
ve = pd.Series(pca.explained_variance_)
ve.plot()
plt.xlabel('Component')
plt.ylabel('Variance Explained')
tmp = pd.DataFrame(pca.components_)
tmp=tmp.iloc[-15:,:]
pca_features=tmp.columns[tmp.abs().max()>0.1]

    
xx= madelonX[:,pca_features]
xx_tst = madelon_tstX[:,pca_features]

## NN testing - standard param set
#grid ={'alpha':nn_reg,'hidden_layer_sizes':nn_arch}
#mlp = MLPClassifier(activation='relu',max_iter=3000,early_stopping=False,random_state=5)
#gs = GridSearchCV(mlp,param_grid=grid,verbose=10,cv=5)
#gs.fit(madelonX[:,pca_features],madelonY)
#print('NN - Standard params - Best CV Score {}'.format(gs.best_score_))
#print('NN - Standard params - Test Score {}'.format(gs.score(xx_tst,madelon_tstY)))
#
#
#
## NN testing - standard param set
#grid ={'alpha':[1e-4,1e-5,1e-6],'hidden_layer_sizes':[(200,100,100,64,100,100,200)]}
#mlp = MLPClassifier(activation='relu',max_iter=3000,early_stopping=False,random_state=5)
#gs = GridSearchCV(mlp,param_grid=grid,verbose=10,cv=5)
#gs.fit(madelonX[:,pca_features],madelonY)
#print('NN - Big network- Best CV Score {}'.format(gs.best_score_))
#print('NN - Big network - Test Score {}'.format(gs.score(xx_tst,madelon_tstY)))


#KNN
knn = KNeighborsClassifier()
grid={'n_neighbors':range(1,25,1),'p':[1,2],'weights':['uniform','distance']}
gs = GridSearchCV(knn,param_grid=grid,cv=5,verbose=10)
gs.fit(xx,madelonY)
print('KNN - Best CV Score {}'.format(gs.best_score_))
print('KNN - Test Score {}'.format(gs.score(xx_tst,madelon_tstY)))


# SVM
dis = pairwise_distances(xx)
m = np.median(dis)
gammas = [(1/m)*x for x in np.arange(0.1,2.1,0.1)]+[0.1,0.2,0.3,0.4,0.5]
gammas = np.arange(0.1,0.9,0.05)

gammas = [(1/m)*x for x in np.arange(0.1,2.1,0.1)]
param_grid={'gamma':gammas,'C':[10**x for x in [-1,0,1,2,3]]}
gs = GridSearchCV(SVC(kernel='rbf',C=1),param_grid=param_grid,cv=5,verbose=10,n_jobs=1)
gs.fit(xx,madelonY)
print('SVM - Best CV Score {}'.format(gs.best_score_))
print('SVM - Test Score {}'.format(gs.score(xx_tst,madelon_tstY)))



