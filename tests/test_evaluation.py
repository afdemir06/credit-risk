from src.model.evaluation import evaluation

def test_evaluation(sample_pipeline,sample_X,sample_y):
    sample_pipeline.fit(sample_X,sample_y)
    result=evaluation(sample_pipeline,sample_X,sample_y)

    assert type(result) is dict
    assert set(["auc_score","precision","recall_score","f1_score"]).issubset(result.keys())
    assert all(0<=v<=1 for v in result.values())