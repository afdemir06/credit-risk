import pytest
import pandas as pd

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "income":[1000,None,3000,2000,None,1500,4500,2500,1500,4000],
        "delay":[1,1,1,1,2,1,3,1,1,2],
        "employment":["working","unemployed",None,"retired","working","working","working","retired",None,"working"]
    })

@pytest.fixture
def sample_filling_dict():
    return {
        "income":"median",
        "delay":"zero",
        "employment":"mode"
    }

@pytest.fixture
def sample_ratio_pairs():
    return [
        ("income","delay")
    ]

@pytest.fixture
def sample_X():
    return pd.DataFrame({
        "income":[1000,4000,3000,2000,5000,1500,2500,3500,4500,7000],
        "delay":[1,1,2,1,2,1,1,2,1,2],
        "employment":["working","unemployed","working","retired","working","working","retired","unemployed","working","retired"]
    })

@pytest.fixture
def sample_y():
    return pd.Series([0,1,0,1,0,1,1,0,0,1])

@pytest.fixture
def sample_pipeline(sample_filling_dict,sample_ratio_pairs):
    from src.model.training import create_pipeline
    return create_pipeline(sample_filling_dict,sample_ratio_pairs)