import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import ResidualsPlot, PredictionError
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError

class PrunedRandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_features='log2', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None,
                 n_pruned_trees=30, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.n_pruned_trees = n_pruned_trees
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            oob_score=True,
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf_.fit(X, y)

        self.oob_scores_ = []
        for i, tree in enumerate(self.rf_.estimators_):
            tree_samples = self.rf_.estimators_samples_[i]
            oob_mask = ~np.isin(np.arange(len(y)), tree_samples)

            if np.any(oob_mask):
                preds = tree.predict(X[oob_mask])
                score = -mean_squared_error(y[oob_mask], preds)
                self.oob_scores_.append(score)
            else:
                self.oob_scores_.append(-np.inf)

        self.oob_scores_ = np.array(self.oob_scores_)
        self.selected_tree_indices_ = np.argsort(self.oob_scores_)[-self.n_pruned_trees:]
        self.pruned_estimators_ = [self.rf_.estimators_[i] for i in self.selected_tree_indices_]

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        preds = np.mean(
            np.array([tree.predict(X) for tree in self.pruned_estimators_]),
            axis=0
        )
        return preds

    def plot_tree_selection(self):
        if not hasattr(self, 'oob_scores_') or not hasattr(self, 'selected_tree_indices_'):
            raise NotFittedError("Model must be fitted before plotting")

        plt.figure(figsize=(14, 7))
        n_trees = len(self.oob_scores_)

        # Plot all trees' OOB scores
        x_all = np.arange(n_trees)
        plt.bar(x_all, self.oob_scores_, color='skyblue', alpha=0.7, label='All Trees')

        # Highlight selected trees
        x_selected = np.array(self.selected_tree_indices_)
        plt.bar(x_selected, self.oob_scores_[x_selected],
                color='crimson', alpha=0.9, label='Selected Trees')

        # Add threshold line
        threshold = np.sort(self.oob_scores_)[-self.n_pruned_trees]
        plt.axhline(y=threshold, color='k', linestyle='--',
                    label=f'Selection Threshold: {threshold:.3f}')

        # Annotate properties
        plt.title(f"Tree Selection Process (Top {self.n_pruned_trees} of {n_trees} Trees)", fontsize=14)
        plt.xlabel("Tree Index", fontsize=12)
        plt.ylabel("OOB Score (Negative MSE)", fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Add text annotations
        plt.text(n_trees*1.02, threshold, f'Threshold: {threshold:.3f}',
                 va='center', ha='left', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()


def compare_rf_models(X_train, y_train, X_test, y_test, random_state=42):
    rf = RandomForestRegressor(n_estimators=100, oob_score=True,
                               random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    prf = PrunedRandomForestRegressor(n_estimators=100, n_pruned_trees=30,
                                     random_state=random_state)
    prf.fit(X_train, y_train)

    # NEW: Plot tree selection process
    prf.plot_tree_selection()

    y_pred_rf = rf.predict(X_test)
    y_pred_prf = prf.predict(X_test)

    # Residual plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    visualizer_rf = ResidualsPlot(rf, ax=axes[0])
    visualizer_rf.fit(X_train, y_train)
    visualizer_rf.score(X_test, y_test)
    axes[0].set_title("Conventional Random Forest Residuals")

    visualizer_prf = ResidualsPlot(prf, ax=axes[1])
    visualizer_prf.fit(X_train, y_train)
    visualizer_prf.score(X_test, y_test)
    axes[1].set_title("Pruned Random Forest Residuals")

    plt.tight_layout()
    plt.show()

    # Prediction error plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    visualizer_rf_pe = PredictionError(rf, ax=axes[0])
    visualizer_rf_pe.fit(X_train, y_train)
    visualizer_rf_pe.score(X_test, y_test)
    axes[0].set_title("Conventional RF Prediction Error")

    visualizer_prf_pe = PredictionError(prf, ax=axes[1])
    visualizer_prf_pe.fit(X_train, y_train)
    visualizer_prf_pe.score(X_test, y_test)
    axes[1].set_title("Pruned RF Prediction Error")

    plt.tight_layout()
    plt.show()

    # Print RMSE
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rmse_prf = np.sqrt(mean_squared_error(y_test, y_pred_prf))

    print(f"Conventional RF RMSE: {rmse_rf:.4f}")
    print(f"Pruned RF RMSE: {rmse_prf:.4f}")


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    compare_rf_models(X_train, y_train, X_test, y_test)