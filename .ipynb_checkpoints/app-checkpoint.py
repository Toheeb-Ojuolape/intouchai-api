from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import huggingface
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain_community import llms
import textwrap
import os
os.environ["HUGGINFACE_API_TOKEN"] = "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"

loader  = TextLoader("data.txt")
document = loader.load()


#Preprocessing
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line,  width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


#Text Splitting
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
docs = text_splitter.split_documents(document)

 
 #Embedding
embeddings = huggingface.HuggingFaceEmbeddings()
db = faiss.FAISS.from_documents(docs, embeddings)

query = "who is Ubaydah?"

doc = db.similarity_search(query)

print(wrap_text_preserve_newlines(str(doc[0].page_content)))

llm = llms.HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8,"max_length":512})

chain = load_qa_chain(llm,chain_type="stuff")

querytext = "Who is Ubaydah?"

docResult = db.similarity_search(querytext)
chain.run(input_documents = docResult, question = querytext)
