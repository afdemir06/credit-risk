from src.data_cleaning import DataCleaningTransformer

def test_no_nulls_after_transform(sample_df, sample_filling_dict):
    cleaner=DataCleaningTransformer(sample_filling_dict)
    result=cleaner.fit_transform(sample_df)
    
    assert result.isnull().sum().sum()==0

def test_median_strategy(sample_df, sample_filling_dict):
    cleaner=DataCleaningTransformer(sample_filling_dict)
    cleaner.fit(sample_df)
    
    assert cleaner.fill_values["income"]==2250.0

def test_zero_strategy(sample_df, sample_filling_dict):
    cleaner=DataCleaningTransformer(sample_filling_dict)
    cleaner.fit(sample_df)

    assert cleaner.fill_values["delay"]==0

def test_mode_strategy(sample_df, sample_filling_dict):
    cleaner=DataCleaningTransformer(sample_filling_dict)
    cleaner.fit(sample_df)

    assert cleaner.fill_values["employment"]=="working"