import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults


# インスタンスの作成
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-1106",
    temperature=0.2,
)

# Web ページからデータを取得します。
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [
    HumanMessage(content="Can LangSmith help test my LLM applications?"),
    AIMessage(content="Yes!"),
]

response = retrieval_chain.invoke(
    {"chat_history": chat_history, "input": "Tell me how"}
)

print(response)
