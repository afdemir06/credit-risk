import logging
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger=logging.getLogger(__name__)

def extract_text(pdf_path: str)->str:
    try:
        logger.info("PDF text extraction started")
        full_text=""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text=page.extract_text()
                if text:
                    full_text+=text+"\n"
            logger.info("PDF extraction done")
            return full_text
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise

def split_text(text: str, chunk_size: int=500, chunk_overlap: int=50)->list:
    try:
        logger.info("Text splitting started")
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks=splitter.split_text(text)
        return chunks
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise

def process_pdf(pdf_path: str, chunk_size: int=500, chunk_overlap: int=50):
    full_text=extract_text(pdf_path)
    chunks=split_text(full_text,chunk_size,chunk_overlap)
    return chunks