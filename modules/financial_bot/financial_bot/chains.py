import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import qdrant_client
from langchain import chains
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from pydantic import PrivateAttr
from unstructured.cleaners.core import (
    clean,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    group_broken_paragraphs,
    replace_unicode_quotes,
)

from financial_bot.embeddings import EmbeddingModelSingleton


class StatelessMemorySequentialChain(chains.SequentialChain):
    """
    A sequential chain that uses a stateless memory to store context between calls.

    This chain overrides the _call and prep_outputs methods to load and clear the memory
    before and after each call, respectively.
    """

    history_input_key: str = "to_load_history"

    def _call(self, inputs: Dict[str, str], **kwargs) -> Dict[str, str]:
        """
        Override _call to load history before calling the chain.

        This method loads the history from the input dictionary and saves it to the
        stateless memory. It then updates the inputs dictionary with the memory values
        and removes the history input key. Finally, it calls the parent _call method
        with the updated inputs and returns the results.
        """

        to_load_history = inputs[self.history_input_key]
        for (
            human,
            ai,
        ) in to_load_history:
            self.memory.save_context(
                inputs={self.memory.input_key: human},
                outputs={self.memory.output_key: ai},
            )
        memory_values = self.memory.load_memory_variables({})
        inputs.update(memory_values)

        del inputs[self.history_input_key]

        return super()._call(inputs, **kwargs)

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """
        Override prep_outputs to clear the internal memory after each call.

        This method calls the parent prep_outputs method to get the results, then
        clears the stateless memory and removes the memory key from the results
        dictionary. It then returns the updated results.
        """

        results = super().prep_outputs(inputs, outputs, return_only_outputs)

        # Clear the internal memory.
        self.memory.clear()
        if self.memory.memory_key in results:
            results[self.memory.memory_key] = ""

        return results


class ContextExtractorChain(Chain):
    """
    Encode the question, search the vector store for top-k articles and return
    context news from documents collection of Alpaca news.

    Attributes:
    -----------
    top_k : int
        The number of top matches to retrieve from the vector store.
    embedding_model : EmbeddingModelSingleton
        The embedding model to use for encoding the question.
    vector_store : qdrant_client.QdrantClient
        The vector store to search for matches.
    vector_collection : str
        The name of the collection to search in the vector store.
    output_key : str
        The key under which the context is returned.
    """

    top_k: int = 1
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    vector_collection: str
    output_key: str = "context"  # Default value for output_key

    def __init__(self, embedding_model, vector_store, vector_collection, top_k, output_key="context"):
        dict={
            "embedding_model": embedding_model,
            "vector_store": vector_store,
            "vector_collection": vector_collection,
            "top_k": top_k,
            "output_key": output_key
        }
        super().__init__(**dict)
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.vector_collection = vector_collection
        self.top_k = top_k
        self.output_key = output_key

    @property
    def input_keys(self) -> List[str]:
        return ["about_me", "question"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        _, quest_key = self.input_keys
        question_str = inputs[quest_key]

        # Clean and encode the question
        cleaned_question = self.clean(question_str)
        cleaned_question = cleaned_question[: self.embedding_model.max_input_length]
        embeddings = self.embedding_model(cleaned_question)

        # Search the vector store
        matches = self.vector_store.search(
            query_vector=embeddings,
            limit=self.top_k,
            collection_name=self.vector_collection,
        )

        # Compile the context from the matches
        context = ""
        for match in matches:
            context += match.payload["summary"] + "\n"

        # Return the context in a dictionary with the specified output key
        return {self.output_key: context}

    def clean(self, question: str) -> str:
        """
        Clean the input question by removing unwanted characters.

        Parameters:
        -----------
        question : str
            The input question to clean.

        Returns:
        --------
        str
            The cleaned question.
        """
        question = clean(question)
        question = replace_unicode_quotes(question)
        question = clean_non_ascii_chars(question)

        return question


class FinancialBotQAChain(Chain):
    """This custom chain handles LLM generation upon given prompt"""

    _lm_function: Any = PrivateAttr()


    def __init__(self, lm_function):
        super().__init__()
        self._lm_function = lm_function

    @property
    def input_keys(self) -> List[str]:
        """Returns a list of input keys for the chain"""

        return ["context"]

    @property
    def output_keys(self) -> List[str]:
        """Returns a list of output keys for the chain"""

        return ["answer"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Calls the chain with the given inputs and returns the output"""

        inputs = self.clean(inputs)
        print(inputs.keys())
        prompt = inputs["about_me"] + inputs["question"] + inputs["context"] + inputs["chat_history"]

        start_time = time.time()
        response = self._lm_function(prompt)
        end_time = time.time()
        duration_milliseconds = (end_time - start_time) * 1000

        return {"answer": response}

    def clean(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Cleans the inputs by removing extra whitespace and grouping broken paragraphs"""

        for key, input in inputs.items():
            cleaned_input = clean_extra_whitespace(input)
            cleaned_input = group_broken_paragraphs(cleaned_input)

            inputs[key] = cleaned_input

        return inputs

class OptimizePromptChain(Chain):
    """This custom chain uses dspy for APE."""

    venv_path = "/home/student/hands-on-llms-13/modules/financial_bot/dspy_env"
    target_directory = "/home/student/hands-on-llms-13/modules/financial_bot/financial_bot"
    command_base = ["python", "dspy_ape.py", "--prompt"]

    @property
    def input_keys(self) -> List[str]:
        return ["about_me", "context", "question"]

    @property
    def output_keys(self) -> List[str]:
        #return ["about_me", "context", "question"]
        return []
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Calls the chain with the given inputs and returns the output"""

        # Convert the dictionary to a JSON string
        prompt = json.dumps({k: inputs[k] for k in self.input_keys})
        command = self.command_base + [prompt]

        # Copy the current environment and modify it for the virtual environment
        copy_current_env = os.environ.copy()
        print(f"VIRTUAL_ENV for dspy: {self.venv_path}")
        copy_current_env["VIRTUAL_ENV"] = self.venv_path
        copy_current_env["PATH"] = os.path.join(self.venv_path, "bin") + os.pathsep + copy_current_env["PATH"]

        result = subprocess.run(
            command,
            cwd=self.target_directory,
            capture_output=True,
            text=True,
            env=copy_current_env,
        )
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        print("Return code:", result.returncode)
        try:
            if result.stdout.strip():  # Check if the output is not empty
                return json.loads(result.stdout)
            else:
                # Handle the case where the output is empty
                raise ValueError("Received empty output from the command.")
        except json.JSONDecodeError as e:
            # Handle JSON decoding errors
            raise ValueError(f"Failed to decode JSON: {e}") from e