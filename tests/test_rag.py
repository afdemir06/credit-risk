import pytest
import os
import tempfile
from fpdf import FPDF
from src.rag.chunking import split_text, extract_text, process_pdf
from src.rag.retrieval import build_query_from_features
from src.rag.generation import build_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_pdf(path: str, text: str = "This is a credit policy document.\nRisk score must be below 0.5.\nDebt-to-income ratio should not exceed 40%."):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(path)


def _ollama_available():
    try:
        import requests
        r = requests.get("http://localhost:11434", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Unit tests — chunking
# ---------------------------------------------------------------------------

class TestSplitText:
    def test_splits_long_text_into_chunks(self):
        text = "word " * 200
        chunks = split_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        assert all(len(c) <= 110 for c in chunks)

    def test_returns_single_chunk_for_short_text(self):
        text = "short text"
        chunks = split_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_empty_text_returns_empty_list(self):
        assert split_text("", chunk_size=100, chunk_overlap=0) == []


class TestExtractText:
    def test_extracts_text_from_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name
        try:
            _create_test_pdf(pdf_path, "Hello Policy World")
            result = extract_text(pdf_path)
            assert "Hello Policy World" in result
        finally:
            os.unlink(pdf_path)

    def test_raises_on_missing_file(self):
        import pytest
        with pytest.raises(Exception):
            extract_text("/nonexistent/file.pdf")


class TestProcessPdf:
    def test_process_pdf_returns_chunks(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name
        try:
            long_text = "policy rule " * 200
            _create_test_pdf(pdf_path, long_text)
            chunks = process_pdf(pdf_path, chunk_size=100, chunk_overlap=10)
            assert len(chunks) >= 1
        finally:
            os.unlink(pdf_path)


# ---------------------------------------------------------------------------
# Unit tests — retrieval
# ---------------------------------------------------------------------------

class TestBuildQueryFromFeatures:
    def test_returns_top_features_joined(self):
        importances = {"income": 0.5, "debt": -0.3, "age": 0.1, "employment": 0.05}
        query = build_query_from_features(importances, top_n=2)
        assert "income" in query
        assert "debt" in query
        assert "employment" not in query
        assert query == "income debt"

    def test_respects_top_n(self):
        importances = {"a": 0.8, "b": 0.6, "c": 0.4, "d": 0.2}
        assert len(build_query_from_features(importances, top_n=3).split()) == 3

    def test_uses_absolute_values_for_sorting(self):
        importances = {"a": 0.2, "b": -0.9, "c": 0.5}
        query = build_query_from_features(importances, top_n=3)
        assert query.split()[0] == "b"

    def test_empty_dict_returns_empty_string(self):
        assert build_query_from_features({}, top_n=5) == ""


# ---------------------------------------------------------------------------
# Unit tests — generation
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    @pytest.fixture
    def sample_chunks(self):
        return [
            ("If debt-to-income exceeds 0.4, flag as high risk.", {"source": "policy.pdf", "chunk_index": 2}, 0.15),
            ("Credit utilization above 80% is not allowed.", {"source": "policy.pdf", "chunk_index": 5}, 0.12),
        ]

    def test_contains_pd_score(self, sample_chunks):
        prompt = build_prompt(78.5, {"income": 50000}, {"income": 0.6}, sample_chunks)
        assert "78.5%" in prompt

    def test_contains_feature_values(self, sample_chunks):
        prompt = build_prompt(50.0, {"income": 50000, "debt": 20000}, {"income": 0.6, "debt": -0.4}, sample_chunks)
        assert "50000" in prompt
        assert "20000" in prompt

    def test_contains_shap_values(self, sample_chunks):
        prompt = build_prompt(50.0, {"income": 50000}, {"income": 0.6123}, sample_chunks)
        assert "0.6123" in prompt

    def test_contains_policy_chunks(self, sample_chunks):
        prompt = build_prompt(50.0, {"income": 50000}, {"income": 0.6}, sample_chunks)
        assert "debt-to-income" in prompt
        assert "Credit utilization" in prompt

    def test_empty_policy_chunks_does_not_crash(self):
        prompt = build_prompt(50.0, {"income": 50000}, {"income": 0.6}, [])
        assert "Probability of Default" in prompt

    def test_empty_feature_importances_does_not_crash(self):
        prompt = build_prompt(50.0, {}, {}, [])
        assert "Probability of Default" in prompt


# ---------------------------------------------------------------------------
# Integration tests — require Ollama running
# ---------------------------------------------------------------------------

ollama = pytest.mark.skipif(not _ollama_available(), reason="Ollama not running at localhost:11434")


@ollama
class TestEmbeddingWithOllama:
    def test_get_embeddings_returns_vectors(self):
        from src.rag.embedding import get_embeddings
        result = get_embeddings(["test text"])
        assert len(result) == 1
        assert len(result[0]) > 0
        assert all(isinstance(v, float) for v in result[0])

    def test_get_embeddings_batch(self):
        from src.rag.embedding import get_embeddings
        result = get_embeddings(["first", "second", "third"])
        assert len(result) == 3


@ollama
class TestStoreAndRetrieve:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        from src.rag.embedding import get_chroma_client, COLLECTION_NAME
        client = get_chroma_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        yield
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    def test_store_then_retrieve(self):
        from src.rag.embedding import store_chunks
        from src.rag.retrieval import retrieve

        chunks = [
            "High debt-to-income ratio increases default risk.",
            "Applicants with stable employment are preferred.",
            "Credit score above 700 is considered good.",
        ]
        store_chunks(chunks, source="test.pdf")

        results = retrieve("debt and income", top_k=2)
        assert len(results) == 2
        texts = [r[0] for r in results]
        assert any("debt" in t.lower() for t in texts)

    def test_retrieve_empty_collection_returns_no_results(self):
        from src.rag.retrieval import retrieve
        results = retrieve("anything", top_k=5)
        assert len(results) == 0


@ollama
class TestGenerateExplanation:
    def test_returns_explanation_string(self):
        from src.rag.generation import generate_explanation
        result = generate_explanation(
            pd_score=85.0,
            feature_values={"income": 30000, "debt": 25000},
            feature_importances={"income": -0.3, "debt": 0.7},
            policy_chunks=[
                ("Debt-to-income ratio above 0.4 is high risk.", {"source": "policy.pdf", "chunk_index": 1}, 0.2)
            ]
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_handles_ollama_unavailable_gracefully(self):
        from src.rag.generation import generate_explanation
        import src.rag.generation as gen
        original_url = gen.OLLAMA_URL
        gen.OLLAMA_URL = "http://localhost:99999/api/generate"
        try:
            result = generate_explanation(50.0, {"x": 1}, {"x": 0.5}, [])
            assert "unavailable" in result.lower() or "unavailable" in result
        finally:
            gen.OLLAMA_URL = original_url
