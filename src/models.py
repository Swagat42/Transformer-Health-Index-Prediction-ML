"""Model definitions and hyperparameter grids (mirrors the notebook)."""
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LassoCV, LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .config import CV_FOLDS, RANDOM_STATE

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False


def get_models():
    """Return {name: (estimator, param_grid or None)}.

    Estimators with a param grid are tuned with GridSearchCV (5-fold, R²).
    Scaling lives inside each pipeline so it is fit on training data only.
    """
    rs = RANDOM_STATE
    models = {
        "Multiple Linear Regression": (LinearRegression(), None),
        "Ridge Regression": (
            make_pipeline(StandardScaler(), RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=CV_FOLDS)),
            None,
        ),
        "Lasso Regression": (
            make_pipeline(StandardScaler(), LassoCV(cv=CV_FOLDS, random_state=rs, max_iter=20000)),
            None,
        ),
        "Elastic Net Regression": (
            make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=rs)),
            None,
        ),
        "Polynomial Regression": (
            make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), LinearRegression()),
            None,
        ),
        "SVR (RBF)": (
            make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)),
            None,
        ),
        "KNN Regression": (
            make_pipeline(StandardScaler(), KNeighborsRegressor()),
            {
                "kneighborsregressor__n_neighbors": [3, 5, 7, 9, 11, 15],
                "kneighborsregressor__weights": ["uniform", "distance"],
                "kneighborsregressor__metric": ["euclidean", "manhattan", "minkowski"],
            },
        ),
        "Decision Tree Regression": (
            DecisionTreeRegressor(random_state=rs),
            {"max_depth": [4, 8, 12, 16, 20], "min_samples_leaf": [1, 5, 10]},
        ),
        "Random Forest Regression": (
            RandomForestRegressor(random_state=rs, n_jobs=-1),
            {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20], "max_features": ["sqrt", "log2", 0.5]},
        ),
        "AdaBoost Regression": (
            AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), random_state=rs),
            {"n_estimators": [50, 100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.5, 1]},
        ),
        "Gradient Boosting Regression": (
            GradientBoostingRegressor(random_state=rs),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [2, 3, 4, 5],
                "subsample": [0.6, 0.8, 1.0],
            },
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost Regression"] = (
            XGBRegressor(objective="reg:squarederror", random_state=rs, n_jobs=-1),
            {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            },
        )
    return models
