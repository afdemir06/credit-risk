from sklearn.base import BaseEstimator,TransformerMixin
import logging

logger=logging.getLogger(__name__)

class DataCleaningTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,fill_strategies: dict):
        self.fill_strategies=fill_strategies
    def fit(self,data,y=None):
        logger.info("Fit process started")
        self.fill_values={}
        try:
            for key,value in self.fill_strategies.items():
                logger.info(f"Column {key} is being fitted with {value}")
                if value=="median":
                    self.fill_values[key]=float(data[key].median())
                elif value=="mode":
                    self.fill_values[key]=data[key].mode()[0]
                else:
                    self.fill_values[key]=0
        except Exception as e:
            logger.error(f"An error occured during fit: {e}")
            raise
        logger.info("Fit process is done")
        return self
    def transform(self,data):
        logger.info("Transform process started")
        c_data=data.copy()
        try:
            for key,value in self.fill_values.items():
                c_data[key]=c_data[key].fillna(value)
            numeric_columns=c_data.select_dtypes(include=["number"]).columns
            c_data[numeric_columns]=c_data[numeric_columns].fillna(c_data[numeric_columns].median())

            categoric_columns=c_data.select_dtypes(exclude=["number"]).columns
            c_data[categoric_columns]=c_data[categoric_columns].fillna("missing")
        except KeyError as e:
            logger.error(f"Column not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during transform: {e}")
            raise
        logger.info("Transform process is done")
        return c_data
    def get_feature_names_out(self,input_features=None):
        return input_features