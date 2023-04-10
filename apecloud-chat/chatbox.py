import sys
import os
import torch
import openai
import argparse
import logging
from langchain import OpenAI
from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_agent, IndexToolConfig, GraphToolConfig
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, QuestionAnswerPrompt
from llama_index import LLMPredictor, PromptHelper, ServiceContext
from llama_index.indices.composability import ComposableGraph
from gpt_index import GPTListIndex, SimpleDirectoryReader
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.embeddings.langchain import LangchainEmbedding
from transformers import pipeline
from read_key import read_key_from_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Query Engine for KubeBlocks.")
    parser.add_argument("query_str", type=str, help="Query string for ask.")
    parser.add_argument("key_file", type=str, help="Key file for OpenAI_API_KEY.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    query_str = args.query_str
    key_file = args.key_file
    print("query:", query_str)

    openai_api_key = read_key_from_file(key_file)
    # set env for OpenAI api key
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # set log level
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))# define LLM

    # initialize the index set with codes and documents
    index_set = {}
    index_set["doc"] = GPTSimpleVectorIndex.load_from_disk('doc.json')
    index_set["code"] = GPTSimpleVectorIndex.load_from_disk('code.json')

    # initialize summary for each index
    index_summaries = ["design and user documents for kubeblocks", "codes of implementations of kubeblocks"]

    # define a LLMPredictor
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

    # define a list index over the vector indices
    # allow us to synthesize information across each index
    graph = ComposableGraph.from_indices(
        GPTListIndex,
        [index_set["doc"], index_set["code"]],
        index_summaries = index_summaries,
        service_context = service_context,
    )

    decompose_transform = DecomposeQueryTransform(
        llm_predictor, verbose=True
    )

    query_configs = [
        {
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs":{
                "similarity_top_k": 1,
            },
            "query_transform": decompose_transform
        },
        {
            "index_struct_type": "list",
            "query_mode": "default",
            "query_kwargs": {
                "response_mode": "tree_summarize",
                "verbose": True
            }
        },
    ]

    # graph config
    graph_config = GraphToolConfig(
        graph = graph,
        name = f"Graph Index",
        description = "useful when you want to answer queries that about how to use and develop with kubeblocks",
        query_configs = query_configs,
        tool_kwargs = {"return_direct": True}
    )

    # define toolkit
    index_configs = []
    for y in range ["doc", "code"]:
        tool_config = IndexToolConfig(
            index = index_set[y],
            name = f"Vectore Index {y}",
            description = f"useful for when you want to answer queries aout the {y} of kubeblocks",
            index_query_kwargs = {"similarity_top_k": 3},
            tool_kwargs = {"retrun_direct": True}
        )
        index_configs.append(tool_config)

    tookit = LlamaToolkit(
        index_configs = index_configs,
        graph_configs = [graph_config]
    )

    # create the llama agent
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0)
    agent_chain = create_llama_agent(
        tookit,
        llm,
        memory = memory,
        verbose = True
    )

    while True:
        text_input = input("User:")
        response = agent_chain.run(input=text_input)
        print(f'Agent: {response}')


if __name__ == "__main__":
    main()

