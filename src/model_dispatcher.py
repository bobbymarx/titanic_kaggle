#model dispatcher.py

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

models= {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "KNN":KNeighborsClassifier(n_neighbors=20),
    "rf": RandomForestClassifier(class_weight='balanced', random_state=42),
    "svc": SVC(C=1),
    "xgb": GradientBoostingClassifier(random_state=42)
}