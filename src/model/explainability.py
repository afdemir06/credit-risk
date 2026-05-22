import shap
import logging
import numpy as np

logger=logging.getLogger(__name__)

def explain_single(pipeline,X):
    try:
        logger.info("Explain single started")   
        X_transformed=pipeline[:-2].transform(X)

        explainer=shap.TreeExplainer(model=pipeline["model"])
        shap_values=explainer(X_transformed)

        feature_names=pipeline[:-2].get_feature_names_out(X.columns.tolist())

        return dict(zip(feature_names,shap_values[0].values))
    except ValueError as e:
        logger.error(f"explain_single expects a single row")
        raise
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise

def explain_batch(pipeline,X):
    try:
        logger.info("Explain batch started")
        X_transformed=pipeline[:-2].transform(X)

        explainer=shap.TreeExplainer(model=pipeline["model"])
        shap_values=explainer(X_transformed)

        feature_names=pipeline[:-2].get_feature_names_out(X.columns.tolist())

        return dict(zip(feature_names,np.abs(shap_values.values).mean(axis=0)))
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise