import logging
import requests
import json
import os

logger=logging.getLogger(__name__)

OLLAMA_URL=os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
MODEL=os.environ.get("OLLAMA_MODEL", "llama3.2")

def build_prompt(
    pd_score: float,
    feature_values: dict,
    feature_importances: dict,
    policy_chunks: list
):
    top_features=sorted(
        feature_importances.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    features_str="\n".join(
        [f"  - {name}: {feature_values.get(name, 'N/A')} (SHAP: {val:.4f})"
         for name, val in top_features]
    )

    policy_str="\n\n".join(
        [f"[Section {meta.get('chunk_index', '?')} from '{meta.get('source', 'policy')}']\n{doc}"
         for doc, meta, _ in policy_chunks]
    )

    prompt=f"""You are a credit risk analyst. Explain why the model made this prediction using the credit policy guidelines.

Probability of Default: {pd_score:.1f}%

Key factors driving this prediction (with SHAP feature importance values):
{features_str}

Relevant credit policy excerpts:
{policy_str}

Explain in clear, simple terms what factors drove this decision. Reference specific policy sections where applicable. Keep the explanation concise (2-4 sentences)."""

    return prompt

def generate_explanation(
    pd_score: float,
    feature_values: dict,
    feature_importances: dict,
    policy_chunks: list
):
    prompt=build_prompt(pd_score, feature_values, feature_importances, policy_chunks)
    logger.info("Calling Ollama for explanation")
    try:
        response=requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        result=response.json()
        explanation=result.get("response", "").strip()
        logger.info("Explanation generated")
        return explanation
    except requests.exceptions.ConnectionError:
        logger.error(f"Ollama not running at {OLLAMA_URL}")
        return "Explanation unavailable — Ollama is not running."
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return f"Explanation unavailable — {str(e)}"
