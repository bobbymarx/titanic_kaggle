#model dispatcher.py

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
models= {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "KNN":KNeighborsClassifier(n_neighbors=20),
    "rf": RandomForestClassifier(),
    "svc": SVC(C=1),
}