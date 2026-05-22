import mlflow
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
import logging

logger=logging.getLogger(__name__)

def evaluation(pipeline,X_test,y_test):
    try:
        logger.info("Evaluation started")
        y_predict_proba=pipeline.predict_proba(X_test)[:,1]
        y_predict=pipeline.predict(X_test)

        metrics={
            "auc_score":roc_auc_score(y_test,y_predict_proba),
            "precision":precision_score(y_test,y_predict),
            "recall_score":recall_score(y_test,y_predict),
            "f1_score":f1_score(y_test,y_predict)
        }
        try:
            mlflow.set_experiment("credit_risk")
            with mlflow.start_run(run_name="evaluation"):
                mlflow.log_metrics(metrics=metrics)
            logger.info("Evaluation is done")
        except Exception as e:
            logger.warning(f"Mlflow logging failed: {e}")
        return metrics
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise