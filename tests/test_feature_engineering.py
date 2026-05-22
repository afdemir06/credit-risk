from src.feature_engineering import FeatureEngineeringTransformer
from src.data_cleaning import DataCleaningTransformer

def test_ratio_pairs(sample_df,sample_ratio_pairs,sample_filling_dict):
    cleaner=DataCleaningTransformer(sample_filling_dict)
    clean_df=cleaner.fit_transform(sample_df)

    feature_eng=FeatureEngineeringTransformer(sample_ratio_pairs)
    result=feature_eng.fit_transform(clean_df)

    assert "income/delay" in result.columns
    assert result["income/delay"].notna().all()