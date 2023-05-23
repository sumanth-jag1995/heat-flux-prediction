import os
import sys
from dataclasses import dataclass

from sklearn.feature_selection import SelectKBest, f_regression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
import logging

from optuna.samplers import TPESampler
from sklearn.metrics import r2_score

import optuna
# Set the log level for the optuna package to WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    feature_selector_file_path = os.path.join("artifacts", "fselector.pkl")

class ModelTrainer:
    def __init__(self, train_array, test_array):
        self.model_trainer_config=ModelTrainerConfig()
        self.X_train = train_array[:,:-1]
        self.y_train = train_array[:,-1]
        self.X_test = test_array[:,:-1]
        self.y_test = test_array[:,-1]
    
    def lgbm_objective(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        num_leaves = trial.suggest_int("num_leaves", 31, 127)
        subsample = trial.suggest_float("subsample", 0.5, 1)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1)

        model = LGBMRegressor(
            n_estimators = n_estimators,
            learning_rate = learning_rate,
            max_depth = max_depth,
            num_leaves = num_leaves,
            subsample = subsample,
            colsample_bytree = colsample_bytree,
            random_state = 42
        )

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        return r2
    
    def catboost_objective(self, trial):
        iterations = trial.suggest_int("iterations", 100, 500)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)
        depth = trial.suggest_int("depth", 3, 10)
        l2_leaf_reg = trial.suggest_int("l2_leaf_reg", 1, 9)
        subsample = trial.suggest_float("subsample", 0.5, 1)

        model = CatBoostRegressor(
            iterations = iterations,
            learning_rate = learning_rate,
            depth = depth,
            l2_leaf_reg = l2_leaf_reg,
            subsample = subsample,
            random_state = 42,
            verbose = 0
        )

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        return r2

    def initiate_model_trainer(self):
        try:
            
            # Run Optuna optimization for each model
            sampler = TPESampler(seed = 42)

            logging.info("Running Optuna optimization for LGBM")
            lgbm_study = optuna.create_study(direction="maximize", sampler=sampler)
            lgbm_study.optimize(self.lgbm_objective, n_trials = 50)

            logging.info("Running Optuna optimization for CatBoost")
            catboost_study = optuna.create_study(direction="maximize", sampler=sampler)
            catboost_study.optimize(self.catboost_objective, n_trials = 50)

            # Train multiple base models with the best parameters
            lgbm = LGBMRegressor(**lgbm_study.best_params, random_state = 42)
            catboost = CatBoostRegressor(**catboost_study.best_params, random_state=42, verbose=0)

            # Feature selection using SelectKBest
            selector = SelectKBest(score_func = f_regression, k = 8)
            X_train_selected = selector.fit_transform(self.X_train, self.y_train)
            X_test_selected = selector.transform(self.X_test)

            # Combine the base models to create an ensemble and assign weights
            weights = [0.3, 0.3]
            # Combine the base models to create an ensemble
            ensemble = VotingRegressor([('lgbm', lgbm), ('catboost', catboost)])
            # Train the ensemble model using selected features
            ensemble.fit(X_train_selected, self.y_train)
            
            logging.info("Saving model to a pickle file")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=ensemble,
            )

            logging.info("Saving selector to a pickle file")
            save_object(
                file_path=self.model_trainer_config.feature_selector_file_path,
                obj=selector,
            )

            predicted = ensemble.predict(X_test_selected)

            r2_square = r2_score(self.y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)