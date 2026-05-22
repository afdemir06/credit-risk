import logging

logger=logging.getLogger(__name__)

def predict_single(pipeline,data):
    try:
        logger.info("Prediction single started")
        return pipeline.predict_proba(data)[:,1]*100
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise

def predict_batch(pipeline,data):
    try:
        logger.info("Prediciton batch started")
        return pipeline.predict_proba(data)[:,1]*100
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise