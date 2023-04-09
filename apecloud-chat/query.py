import sys
import os
import torch
import openai
import argparse
import logging
from langchain import OpenAI
from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, QuestionAnswerPrompt, LLMPredictor, PromptHelper, ServiceContext
from gpt_index import GPTListIndex, SimpleDirectoryReader
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.embeddings.langchain import LangchainEmbedding
from transformers import pipeline
from IPython.display import Markdown, display

def parse_arguments():
    parser = argparse.ArgumentParser(description="Query Engine for KubeBlocks.")
    parser.add_argument("query_str", type=str, help="Query string for ask.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    query_str = args.query_str
    print("query:", query_str)

    # set env for OpenAI api key
    os.environ['OPENAI_API_KEY'] = "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"

    # set log level
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))# define LLM

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))


    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 4096
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # load from disk
    index1 = GPTSimpleVectorIndex.load_from_disk('doc.json')
    index2 = GPTSimpleVectorIndex.load_from_disk('code.json')

    response = index1.query(query_str)
    print("answer:", response)
    response = index2.query(query_str)
    print("answer:", response)


if __name__ == "__main__":
    main()

