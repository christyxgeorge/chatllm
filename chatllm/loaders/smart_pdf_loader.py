"""Smart PDF Loader"""
import logging

from typing import Any, Dict, List, Optional

from chatllm.loaders.base import BaseReader, Document

LLMSHERPA_API_URL = (
    "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
)

logger = logging.getLogger(__name__)


class SmartPDFLoader(BaseReader):
    """
    SmartPDFLoader uses nested layout information such as sections, paragraphs, lists
    and tables to smartly chunk PDFs for optimal usage of LLM context window

    Args:
        llmsherpa_api_url (str): Address of the service hosting llmsherpa PDF parser
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from llmsherpa.readers import LayoutPDFReader

        self.pdf_reader = LayoutPDFReader(LLMSHERPA_API_URL)

    def load_data(self, pdf_path_or_url: str, extra_info: Optional[Dict] = None) -> List[Document]:
        """Load data and extract table from PDF file.

        Args:
            pdf_path_or_url (str): A url or file path pointing to the PDF

        Returns:
            List[Document]: List of documents.
        """
        results = []
        total_chars = 0
        doc = self.pdf_reader.read_pdf(pdf_path_or_url)
        for chunk in doc.chunks():
            text = chunk.to_context_text()
            document = Document(
                text=text,
                extra_info={
                    "processor": "llmsherpa",
                    "chunk_type": chunk.tag,
                    "page_idx": chunk.page_idx,
                    "file_name": pdf_path_or_url,
                    "level": chunk.level,
                },
            )
            results.append(document)
            total_chars = total_chars + len(text)
            # logger.debug(f"Added document => {len(text)} bytes, Doc Count = {len(results)}")
        logger.info(
            f"Loaded file: {pdf_path_or_url} => [{total_chars} bytes, {len(results)} documents]"
        )
        return results
