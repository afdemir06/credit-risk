from sklearn.base import BaseEstimator,TransformerMixin
import logging
import numpy as np

logger=logging.getLogger(__name__)

class FeatureEngineeringTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,ratio_pairs: list=[]):
        self.ratio_pairs=ratio_pairs
    def fit(self,data,y=None):
        return self
    def transform(self,data):
        logger.info("Transform process started")
        c_data=data.copy()
        if self.ratio_pairs:
            try:
                for i in self.ratio_pairs:
                    c_data[f"{i[0]}/{i[1]}"]=c_data[i[0]]/(c_data[i[1]]+1e-8)
            except KeyError as e:
                logger.error(f"Column not found: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        logger.info("Transform process is done")
        logger.info(f"Columns: {c_data.columns.tolist()}")
        return c_data
    def get_feature_names_out(self,input_features):
        feature_names=list(input_features)
        if self.ratio_pairs:
            for pair in self.ratio_pairs:
                feature_names.append(f"{pair[0]}/{pair[1]}")
        return np.array(feature_names)