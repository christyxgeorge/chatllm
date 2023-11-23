"""Smart PDF Loader"""
from typing import Any, Dict, List, Optional

from chatllm.loaders.base import BaseReader, Document

LLMSHERPA_API_URL = (
    "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
)


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
        doc = self.pdf_reader.read_pdf(pdf_path_or_url)
        for chunk in doc.chunks():
            document = Document(
                text=chunk.to_context_text(),
                extra_info={
                    "processor": "llmsherpa",
                    "chunk_type": chunk.tag,
                    "file_name": pdf_path_or_url,
                },
            )
            results.append(document)
        return results
