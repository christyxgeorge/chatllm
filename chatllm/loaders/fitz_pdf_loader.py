"""PDF Loader"""
import logging
import re

from typing import Any, Dict, List, Optional

import fitz

from chatllm.loaders.base import BaseReader, Document

logger = logging.getLogger(__name__)


class FitzPDFLoader(BaseReader):
    """
    FitzPDFLoader uses PyMuPDF to chunk PDFs for optimal usage of LLM context window
    """

    def __init__(self, *args, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def preprocess(self, text) -> str:
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text

    def load_data(self, pdf_path_or_url: str, extra_info: Optional[Dict] = None) -> List[Document]:
        """Load data and extract table from PDF file.

        Args:
            pdf_path_or_url (str): A url or file path pointing to the PDF

        Returns:
            List[Document]: List of documents.
        """
        results = []
        doc = fitz.Document(pdf_path_or_url)
        logger.debug(f"Number of pages: {doc.page_count} // {doc.metadata}")
        for i, page in enumerate(doc):
            doc_page = page.get_text()
            logger.debug(f"== Doc Page {i+1}: {len(doc_page)} chars => {doc_page}")
            processed_doc = self.preprocess(doc_page)
            document = Document(
                text=processed_doc,
                extra_info={"processor": "fitz", "file_name": pdf_path_or_url},
            )
            results.append(document)
        return results
