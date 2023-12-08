import json
import logging
import os

from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

import chromadb

from chromadb.config import Settings
from pydantic import BaseModel

from chatllm.llm_params import LLMConfig, LLMParam
from chatllm.loaders import SmartPDFLoader
from chatllm.prompts import (
    ChatMessage,
    ChatPromptValue,
    ChatRole,
    PromptValue,
    StringPromptValue,
)
from chatllm.prompts.default_prompts import simple_system_prompt

# from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class LLMHistoryItem(BaseModel):
    text: str
    role: ChatRole


class LLMSession:
    def __init__(self, llm, model_name, model_cfg) -> None:
        self.llm = llm
        self.model_name = model_name
        self.model_config: LLMConfig | None = model_cfg
        self.system_prompt_type = "simple"
        self.system_prompt = simple_system_prompt
        self.chat_history: list[LLMHistoryItem] = []
        self.files: List[str] = []
        # self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Create an in-memory ChromaDB client (Currently, set anonymized_telemetry to False)
        chromadb_anonymized_telemetry = os.environ.get("CHROMADB_ANONYMIZED_TELEMETRY", False)
        self.chroma_client = chromadb.Client(
            Settings(anonymized_telemetry=chromadb_anonymized_telemetry)
        )
        self.index_name = "session_idx"
        self.collection = None

    def get_model_params(self) -> Dict[str, LLMParam]:
        assert (  # nosec  # noqa: S101
            self.model_config is not None
        ), f"Model {self.model_name} not loaded"
        return self.model_config.get_params()

    def add_history(self, text: str, role: ChatRole) -> None:
        self.chat_history.append(LLMHistoryItem(text=text, role=role))

    def clear(self) -> None:
        """clear the chat history"""
        self.chat_history = []
        self.files = []
        if self.collection:
            self.chroma_client.delete_collection(self.index_name)

    def get_document_count(self) -> None:
        """Show the number of documents in the session index"""
        return self.collection.count() if self.collection else 0

    def list_indexes(self) -> None:
        """List collections within the chroma client"""
        return self.chroma_client.list_collections()

    def query_index(self, query_string) -> Dict[str, List[Any]] | None:
        """Query the index"""
        qresult = self.collection.query(query_texts=[query_string], n_results=10)
        if qresult["documents"]:
            return {"documents": qresult["documents"][0], "metadatas": qresult["metadatas"][0]}
        else:
            return None

    def set_system_prompt(self, type: str, prompt: str = "") -> None:
        """Set the system prompt"""
        if prompt:
            self.system_prompt_type = type
            self.system_prompt = prompt
        else:
            self.system_prompt_type = "none"
            self.system_prompt = ""

    def create_prompt_value(self, user_query, summarize=False) -> PromptValue:
        """Create a PromptValue object for Document Querying or Chatting"""
        if summarize:
            user_query = self.create_summarization_prompt(user_query)
        elif self.files:
            user_query = self.create_indexed_prompt(user_query)

        if self.system_prompt:
            prompt_value = ChatPromptValue()
            if self.system_prompt or self.chat_history:
                prompt_value.add_message(
                    ChatMessage(role=ChatRole.SYSTEM, content=self.system_prompt)
                )
            for msg in self.chat_history:
                prompt_value.add_message(ChatMessage(role=msg.role, content=msg.text))
            # Add the User Query
            prompt_value.add_message(ChatMessage(role=ChatRole.USER, content=user_query))
            return prompt_value
        else:
            sprompt_value = StringPromptValue(text=user_query)
            return sprompt_value

    def create_indexed_prompt(self, user_query):
        documents = self.query_index(user_query)
        if documents:
            # TODO: Need to handle docs longer than the context length
            # context_length = self.model_config.max_context_length
            logger.info(f"{len(documents['documents'])} documents found for query: {user_query}")
            prompt_prefix = "Given the following documents:"
            index_docs = "\n\n".join([doc for doc in documents["documents"]])
            prompt_suffix = "\n Please answer the following question:"
            final_prompt = f"{prompt_prefix}\n{index_docs}\n{prompt_suffix}\n{user_query}"
        else:
            # TODO: Need to return a warning to the user.
            logger.info(f"No documents found for query: {user_query}")
            final_prompt = user_query
        return final_prompt

    def create_summarization_prompt(self, user_query):
        documents = self.collection.get()
        # TODO: Need to handle docs longer than the context length
        # context_length = self.model_config.max_context_length
        # logger.info(f"{len(documents['documents'])} document chunks found")
        prompt_prefix = "Given the following documents"
        index_docs = "\n\n".join([doc for doc in documents["documents"]])
        final_prompt = f"{prompt_prefix}\n{index_docs}\n{user_query}"
        return final_prompt

    def add_file(self, file_name: str) -> None:
        """Add a file to the session"""
        mpath = Path(file_name)
        if not mpath.is_file():
            raise FileNotFoundError(f"File {file_name} does not exist")
        self.files.append(file_name)

        self.index_file(file_name)

    def index_file(self, file_name: str) -> None:
        """Index a file, Create index if not created"""
        if not self.collection:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.index_name,
                # metadata={"hnsw:space": "cosine"},
                #    Hnswlib: Valid options for hnsw:space are "l2", "ip, "or "cosine"
                # embedding_function=default_ef
            )
        else:
            logger.info(f"Number of documents in the index: {self.collection.count()}")

        # documents = FitzPDFLoader().load_data(file_name)
        smart_docs = SmartPDFLoader().load_data(file_name)

        texts = [doc.text for doc in smart_docs]
        metadata = [doc.metadata for doc in smart_docs]
        ids = [f"{file_name}.{i+1}" for i in range(len(smart_docs))]  # TODO: Can we do MD5?
        # embeddings = self.embedding_model.encode([doc.text for doc in smart_docs])
        # self.collection.add(embeddings=embeddings, documents=texts, metadatas=metadata, ids=ids)
        self.collection.add(documents=texts, metadatas=metadata, ids=ids)
        logger.info(f"Number of documents in the index: {self.collection.count()}")

    async def run_stream(
        self,
        user_query: str,
        verbose=False,
        word_by_word=False,
        **kwargs,
    ) -> AsyncGenerator[Any | str, Any]:
        assert self.llm is not None, f"Model {self.model_name} not loaded"  # nosec  # noqa: S101

        # TODO: Need to move this out to the CLI/Gradio layer?
        if verbose:
            logger.info("=" * 50)
            logger.info(f"Prompt = {user_query}")
            logger.info(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            summarize = kwargs.pop("summarize") if "summarize" in kwargs else False
            if summarize and not user_query:
                user_query = "Please summarize in a concise manner"
            prompt_value: PromptValue = self.create_prompt_value(user_query, summarize=summarize)
            self.add_history(user_query, role=ChatRole.USER)
            stream = await self.llm.generate_stream(prompt_value, verbose=verbose, **kwargs)
            async for llm_response in stream:
                response_text = (
                    llm_response.get_first_of_last_token()
                    if word_by_word
                    else llm_response.get_first_sequence()
                )
                if response_text:
                    yield "content", response_text
                elif llm_response.finish_reasons:
                    finish_reasons = "|".join(llm_response.finish_reasons)
                    yield "warning", f"No response from LLM [Reason = {finish_reasons}]"

            yield "done", ""
            self.add_history(llm_response.get_first_sequence(), role=ChatRole.AI)
            if verbose:
                llm_response.print_summary()

        except Exception as e:
            logger.warning(f"Exception = {e}")
            # TODO: Can't do gradio specific in this class!
            yield "error", f"Unable to generate response [{e}]"

    async def run_batch(
        self,
        user_query: str,
        verbose=False,
        **kwargs,
    ) -> Any:
        assert self.llm is not None, f"Model {self.model_name} not loaded"  # nosec  # noqa: S101
        if verbose:
            logger.info("=" * 50)
            logger.info(f"User Query = {user_query}")
            logger.info(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            summarize = kwargs.pop("summarize") if "summarize" in kwargs else False
            if summarize and not user_query:
                user_query = "Please summarize in a concise manner"
            prompt_value: PromptValue = self.create_prompt_value(user_query, summarize=summarize)
            self.add_history(user_query, role=ChatRole.USER)
            llm_response = await self.llm.generate(prompt_value, verbose=verbose, **kwargs)
            response_text = llm_response.get_first_sequence()
            if not response_text:
                yield "warning", "No response from LLM"
            else:
                yield "content", response_text

            yield "done", ""
            self.add_history(llm_response.get_first_sequence(), role=ChatRole.AI)
            if verbose:
                llm_response.print_summary()

        except Exception as e:
            logger.warning(f"Exception = {e}")
            yield "error", f"Unable to generate response [{e}]"
