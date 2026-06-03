import logging
from src.rag.embedding import get_embeddings, get_or_create_collection

logger=logging.getLogger(__name__)

def retrieve(query: str, top_k: int=5):
    collection=get_or_create_collection()
    query_embedding=get_embeddings([query])[0]
    results=collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    documents=results.get("documents", [[]])[0]
    metadatas=results.get("metadatas", [[]])[0]
    distances=results.get("distances", [[]])[0]
    logger.info(f"Retrieved {len(documents)} chunks for query")
    return list(zip(documents, metadatas, distances))

def build_query_from_features(feature_importances: dict, top_n: int=5):
    sorted_features=sorted(
        feature_importances.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    top_features=[f[0] for f in sorted_features[:top_n]]
    query=" ".join(top_features)
    logger.info(f"Query from top features: {query}")
    return query
