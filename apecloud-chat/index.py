from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, QuestionAnswerPrompt, LLMPredictor, PromptHelper, ServiceContext, Document
import logging
import sys
import os
import torch
import openai
from gpt_index import GPTListIndex, SimpleDirectoryReader
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from transformers import pipeline
from IPython.display import Markdown, display
from traverse_code_files import  list_go_files_recursive
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter


# set env for OpenAI api key
os.environ['OPENAI_API_KEY'] = "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"

# set log level
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))# define LLM

# define the LLM predictor
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

text_splitter = TokenTextSplitter(separator=" ", chunk_size=2048, chunk_overlap=20)
embed_model = OpenAIEmbedding()

code_path = "/root/kubeblocks/"
go_files = list_go_files_recursive(code_path)

documents = []
for go_file in go_files:
    print(go_file)
    docs = SimpleDirectoryReader(input_files=[go_file]).load_data()
    if len(docs) > 0:
        documents.append(docs[0])
'''
    if len(documents) > 0:
        #GPTSimpleVectorIndex.from_documents(documents)
        document = documents[0]
        text_chunks = text_splitter.split_text(document.text)
        doc_chunks = [Document(t) for t in text_chunks]
        # insert new document chunks
        for doc_chunk in doc_chunks:
            print(doc_chunk.text)
            #print(doc_chunk.get_embedding())
            index.insert(doc_chunk)
        index.refresh([])
'''

index = GPTSimpleVectorIndex.from_documents(documents)

index.save_to_disk('code.json')

#documents = SimpleDirectoryReader(data_path, recursive=True).load_data()
#response = index.query("what is kubeblocks")
#print(response)
#display(Markdown(f"<b>{response}</b>"))
