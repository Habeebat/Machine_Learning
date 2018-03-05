#k-fold cross validation
import pandas
from sklearn import tree
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn import grid_search

dataset = pandas.read_csv("wine_data.csv",names=["Class","Alcohol","Malic Alic","Ash","Alcanility of Ash","Magnesium",
                    "Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280_OD315_diluted wines",
                    "Proline"])

y = dataset["Class"].values
del(dataset["Class"])

########################################################################################

clh = tree.DecisionTreeClassifier(max_depth  = 10)
parameters = {'max_depth':np.arange(1,20)}
clfx = grid_search.GridSearchCV(clh, parameters)
clfx.fit(dataset.values, y)
print(clfx.best_params_)

########################################################################################


clh = tree.DecisionTreeClassifier(max_depth  = 4)
clf = AdaBoostClassifier(base_estimator= clh,n_estimators=10)
clf.fit(dataset.values, y)


########################################################################################


print(np.mean(cross_validation.cross_val_score(clf, dataset.values, y, cv=4)))
