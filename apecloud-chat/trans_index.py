from langchain import OpenAI
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, QuestionAnswerPrompt, LLMPredictor, PromptHelper, ServiceContext, Document
import logging
import sys
import os
import argparse
from gpt_index import GPTListIndex, SimpleDirectoryReader
from traverse_code_files import  list_go_files_recursive
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from read_key import read_key_from_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Query Engine for KubeBlocks.")
    parser.add_argument("key_file", type=str, help="Key file for OpenAI_API_KEY.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    key_file = args.key_file

    openai_api_key = read_key_from_file(key_file)
    # set env for OpenAI api key
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # set log level
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    if True:
        index = GPTSimpleVectorIndex.load_from_disk('doc.json')

    if False:
        documents = SimpleDirectoryReader(doc_path, required_exts=[".yaml"], recursive=True).load_data()
        index = GPTSimpleVectorIndex.from_documents(documents)
        index.save_to_disk('config.json')

    if False:
        code_path = "/root/kubeblocks/"
        go_files = list_go_files_recursive(code_path)

        documents = []
        for go_file in go_files:
            print(go_file)
            docs = SimpleDirectoryReader(input_files=[go_file]).load_data()
            if len(docs) > 0:
                documents.append(docs[0])

        index = GPTSimpleVectorIndex.from_documents(documents)
        index.save_to_disk('code.json')

if __name__ == "__main__":
    main()
