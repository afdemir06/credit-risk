import joblib
import logging

logger=logging.getLogger(__name__)

def save_model(pipeline,path):
    try:
        logger.info("The model saving...")
        joblib.dump(pipeline,path)
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise
def load_model(path):
    try:
        logger.info("Model loading...")
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise

def download_results(data):
    try:
        logger.info("Results downloading...")
        return data.to_csv(index=False).encode("utf-8")
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise