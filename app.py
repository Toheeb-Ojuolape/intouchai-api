from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import huggingface
from langchain_community.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain_community import llms
import textwrap
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

loader = TextLoader("data.txt")
document = loader.load()

# Preprocessing
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

# Text Splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)

# Embedding
embeddings = huggingface.HuggingFaceEmbeddings()
db = faiss.FAISS.from_documents(docs, embeddings)

query = "What is my name?"

doc = db.similarity_search(query)

# Use llms.HuggingFaceHub to wrap the model
llm = llms.HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})

chain = load_qa_chain(llm, chain_type="stuff")

querytext = "What is my name?"

docResult = db.similarity_search(querytext)
response = chain.invoke({'input_documents': docResult, 'question': query})

print(response['output_text'])
