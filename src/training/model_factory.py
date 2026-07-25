"""Model candidates used by the training command."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def candidates(random_state: int = 42):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "svm": SVC(probability=True, class_weight="balanced", random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=9),
        "gaussian_nb": GaussianNB(),
    }
