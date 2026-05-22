from src.model.training import train

def test_train(sample_X,sample_y,sample_pipeline,sample_filling_dict,sample_ratio_pairs,n_trials=1):
    result=train(sample_X,sample_y,sample_pipeline,sample_filling_dict,sample_ratio_pairs,n_trials)

    assert result is not None
    assert result.predict_proba(sample_X).shape[0]==len(sample_X)