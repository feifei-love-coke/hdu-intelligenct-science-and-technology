from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime

data = load_breast_cancer()
X = data.data
y = data.target

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=42)

Kernel = ["linear","poly","rbf","sigmoid"]
for kernel in Kernel:
    time0 = time()
    clf= SVC(kernel = kernel
             , gamma="auto"
             , degree = 1
             , cache_size=5000
            ).fit(Xtrain,Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel,clf.score(Xtest,Ytest)))
    print(time()-time0)

import pandas as pd
data = pd.DataFrame(X)
data.describe( ).T#描述性统计
#从mean列和std列可以看出严重的量纲不统一
#这个时候就需要对数据进行标准化

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)#将数据转化为0,1正态分布
data = pd.DataFrame(X)
data.describe( ).T#均值很接近，方差为1了

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,y,test_size=0.3,random_state=42)
Kernel = ["linear","poly","rbf","sigmoid"]
for kernel in Kernel:
    time0 = time()
    clf= SVC(kernel = kernel
             , gamma="auto"
             , degree = 1
             , cache_size=5000
            ).fit(Xtrain,Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel,clf.score(Xtest,Ytest)))
    print(time()-time0)

score = []
gamma_range = np.logspace(-10, 1, 50) #返回在对数刻度上均匀间隔的数字
for i in gamma_range:
    clf = SVC(kernel="rbf",gamma = i,cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
    
print(max(score), gamma_range[score.index(max(score))])
plt.plot(gamma_range,score)
plt.show()

clf = SVC(kernel="rbf",gamma = 0.004291934260128779,cache_size=5000).fit(Xtrain,Ytrain)
score = clf.score(Xtrain,Ytrain)
print(score)

from sklearn.model_selection import cross_val_score
clf = SVC(kernel="rbf",gamma = 0.004291934260128779,cache_size=5000)
score_cv = cross_val_score(clf,Xtrain,Ytrain,cv=10).mean()
print(score_cv)

from sklearn.model_selection import StratifiedShuffleSplit#用于支持带交叉验证的网格搜索
from sklearn.model_selection import GridSearchCV#带交叉验证的网格搜索
time0 = time()
gamma_range = np.logspace(-10,1,20)
coef0_range = np.linspace(0,5,10)
param_grid = dict(gamma = gamma_range
                  ,coef0 = coef0_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)#将数据分为5份，5份数据中测试集占30%
grid = GridSearchCV(SVC(kernel = "poly",degree=1,cache_size=5000)
                    ,param_grid=param_grid
                    ,cv=cv)
grid.fit(X, y)
print("The best parameters are %s with a score of %0.5f" % (grid.best_params_, 
grid.best_score_))
print(time()-time0)

