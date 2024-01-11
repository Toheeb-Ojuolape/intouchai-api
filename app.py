from langchain_community.document_loaders import TextLoader

loader = TextLoader('data.txt')
document = loader.load()
print(document)

#Preprocessing 

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')    

    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

print(wrap_text_preserve_newlines(str(document[0])))