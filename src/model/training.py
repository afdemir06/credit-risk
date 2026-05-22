import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_cleaning import DataCleaningTransformer
from feature_engineering import FeatureEngineeringTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer,make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import logging
import mlflow
import optuna

logger=logging.getLogger(__name__)

def create_pipeline(fill_strategies,ratio_pairs):
    if isinstance(fill_strategies,str):
        import json
        fill_strategies=json.loads(fill_strategies)
    if isinstance(ratio_pairs,str):
        ratio_pairs=json.loads(ratio_pairs)
    try:
        logger.info("Pipeline started to be created")
        preprocessing=ColumnTransformer(
            transformers=[
                ("num","passthrough",selector(dtype_include="number")),
                ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),selector(dtype_include=["string","object","category"]))
            ],
            remainder="passthrough"
        )

        pipeline=ImbPipeline(
            steps=[
                ("cleaning",DataCleaningTransformer(fill_strategies)),
                ("engineering",FeatureEngineeringTransformer(ratio_pairs)),
                ("preprocessing",preprocessing),
                ("smote",SMOTE(random_state=42)),
                ("model",XGBClassifier())
            ]
        )
        logger.info("Pipeline created")
        return pipeline
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise

def train(X_train,y_train,pipeline,fill_strategies,ratio_pairs,n_trials=50):
    try:
        logger.info("Train started")
        def objective(trial):
            params={
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1)
            }
            trial_pipeline=create_pipeline(fill_strategies,ratio_pairs)
            trial_pipeline.named_steps["model"].set_params(**params)
            scores=cross_val_score(trial_pipeline,X_train,y_train,cv=5,scoring="roc_auc")
            return scores.mean()
        def optuna_logging_callback(study,trial):
            logger.info(f"Trial {trial.number} finished-AUC: {trial.value:.4f}")
    
        study=optuna.create_study(direction="maximize")
        study.optimize(objective,n_trials=n_trials,callbacks=[optuna_logging_callback])

        pipeline.named_steps["model"].set_params(**study.best_params)
        pipeline.fit(X_train,y_train)
        try:
            mlflow.set_experiment("credit_risk")
            with mlflow.start_run():
                mlflow.log_params(study.best_params)
                mlflow.log_metric("auc_roc",study.best_trial.value)
                mlflow.sklearn.log_model(pipeline,"model")
        except Exception as e:
            logger.warning(f"Mlflow logging failed: {e}")
        logger.info("Train finished")
        return pipeline
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise