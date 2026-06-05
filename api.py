from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd
import io
import json
import os
from src.model import training,evaluation,prediction,explainability
from src.rag import chunking, embedding, retrieval, generation
from src import utils
from sklearn.model_selection import train_test_split
import logging
from src.config import MODELS_DIR, POLICIES_DIR
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s"
)

logger=logging.getLogger(__name__)

class PredictSingleRequest(BaseModel):
    data: dict

policy_loaded=False

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri("http://mlflow:5000")
    global pipeline, policy_loaded
    model_path=MODELS_DIR/"model.joblib"
    if model_path.exists():
        pipeline=utils.load_model(model_path)
        logger.info("Model loaded")
    else:
        pipeline=None
        logger.warning("Model not found")
    if embedding.collection_count() > 0:
        policy_loaded=True
        logger.info("Policy data found in ChromaDB")
    yield

app=FastAPI(lifespan=lifespan)

@app.post("/train")
async def train(
    file: UploadFile=File(...),
    fill_strategies: str=Form(...),
    ratio_pairs: str=Form(default="[]"),
    target_column: str=Form(...)
):
    try:
        global pipeline
        logger.info("Train file has been taken")
        temp_file_path=f"temp_{file.filename}"
        with open(temp_file_path,"wb") as f:
            content=await file.read()
            if not content:
                raise ValueError("File is empty")
            f.write(content)
        try:
            df=pd.read_csv(temp_file_path) if file.filename.endswith(".csv") else pd.read_excel(temp_file_path)
        except Exception as e:
            raise ValueError("File not read")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        logger.info(f"Columns: {df.columns.tolist()}")
        X=df.drop(columns=[target_column])
        y=df[target_column]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

        strategies=json.loads(fill_strategies)
        if isinstance(strategies,str):
            strategies=json.loads(strategies)
        pairs=json.loads(ratio_pairs)
        if isinstance(pairs,str):
            pairs=json.loads(pairs)

        pipeline=training.create_pipeline(strategies,pairs)
        pipeline=training.train(X_train,y_train,pipeline,strategies,pairs)
        MODELS_DIR.mkdir(exist_ok=True)
        utils.save_model(pipeline,MODELS_DIR/"model.joblib")
        with open(MODELS_DIR/"features.json","w") as f:
            json.dump(X.columns.tolist(),f,indent=4)

        metrics=evaluation.evaluation(pipeline,X_test,y_test)
        with open(MODELS_DIR/"metrics.json","w") as f:
            json.dump(metrics,f,indent=4)

        with open(MODELS_DIR/"target_column.json","w") as f:
            json.dump(target_column,f,indent=4)

        logger.info("Metrics returned")
        return {"status":"success"}
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise HTTPException(status_code=500,detail=f"Error occured: {e}")

@app.post("/predict/single")
async def predict_single(request: PredictSingleRequest):
    try:
        logger.info("Single predict data has been taken")
        global pipeline
        df=pd.DataFrame([request.data])
        predict_proba=prediction.predict_single(pipeline,df)
        feature_importances=explainability.explain_single(pipeline,df)
        feature_importances={k: float(v) for k,v in feature_importances.items()}

        result_dict={
            "predict_proba":predict_proba.tolist(),
            "feature_importances":feature_importances
        }
        logger.info("Explain data returned")
        return {"status":"success","results":result_dict}
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise HTTPException(status_code=500,detail=f"Error occured: {e}")
    
@app.post("/predict/batch")
async def predict_batch(file: UploadFile=File(...)):
    try:
        logger.info("Batch prediction data has been taken")
        global pipeline
        contents=await file.read()
        df=pd.read_csv(io.BytesIO(contents)) if file.filename.endswith(".csv") else pd.read_excel(io.BytesIO(contents))
        predict_proba=prediction.predict_batch(pipeline,df)
        feature_importances=explainability.explain_batch(pipeline,df)
        feature_importances={k: float(v) for k,v in feature_importances.items()}

        result_dict={
            "predict_proba":predict_proba.tolist(),
            "feature_importances":feature_importances
        }
        logger.info("Explain data returned")
        return {"status":"success","results":result_dict}
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise HTTPException(status_code=500,detail=f"Error occured: {e}")
    
@app.post("/policy/upload")
async def upload_policy(file: UploadFile=File(...)):
    try:
        logger.info("Policy PDF upload started")
        temp_path=f"temp_{file.filename}"
        with open(temp_path,"wb") as f:
            content=await file.read()
            if not content:
                raise ValueError("File is empty")
            f.write(content)
        try:
            chunks=chunking.process_pdf(temp_path)
            embedding.store_chunks(chunks, file.filename)
            global policy_loaded
            policy_loaded=True
            logger.info(f"Policy '{file.filename}' processed ({len(chunks)} chunks)")
            return {"status":"success","chunks":len(chunks),"filename":file.filename}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        logger.error(f"Error processing policy: {e}")
        raise HTTPException(status_code=500,detail=f"Error processing policy: {e}")

@app.post("/predict/explain")
async def predict_explain(request: PredictSingleRequest):
    try:
        global pipeline, policy_loaded
        if pipeline is None:
            raise HTTPException(status_code=400, detail="No model trained yet")
        if not policy_loaded:
            raise HTTPException(status_code=400, detail="No policy uploaded yet")
        logger.info("Explain prediction started")
        df=pd.DataFrame([request.data])
        predict_proba=prediction.predict_single(pipeline,df)
        feature_importances=explainability.explain_single(pipeline,df)
        feature_importances={k: float(v) for k, v in feature_importances.items()}
        query=retrieval.build_query_from_features(feature_importances)
        policy_chunks=retrieval.retrieve(query, top_k=5)
        explanation=generation.generate_explanation(
            pd_score=float(predict_proba[0]),
            feature_values=request.data,
            feature_importances=feature_importances,
            policy_chunks=policy_chunks
        )
        return {
            "status":"success",
            "results":{
                "predict_proba":predict_proba.tolist(),
                "feature_importances":feature_importances,
                "explanation":explanation
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during explain prediction: {e}")
        raise HTTPException(status_code=500,detail=f"Error: {e}")

@app.get("/model/info")
async def model_info():
    info_path=MODELS_DIR/"features.json"
    if pipeline is None or not info_path.exists():
        return {"model_exist":False,"features":None}
    with open(info_path,"r") as f:
        features=json.load(f)
    with open(MODELS_DIR/"metrics.json","r") as f:
        metrics=json.load(f)
    with open(MODELS_DIR/"target_column.json","r") as f:
        target_column=json.load(f)
    global policy_loaded
    return {
        "model_exist":True,
        "features":features,
        "metrics":metrics,
        "target_column":target_column,
        "policy_loaded":policy_loaded or embedding.collection_count()
        }