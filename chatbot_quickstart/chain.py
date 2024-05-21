import os


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Dict

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch


# インスタンスの作成
chat = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-1106",
    temperature=0.2,
)


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


# --Retrievers-- リトリーバは、文字列クエリを入力として受け取り、ドキュメントのリストを 出力として返す。
# Web ページからデータを取得します。
loader = WebBaseLoader("https://skysthelimit.co.jp/about.html")
data = loader.load()

# LLMのコンテキスト・ウィンドウが扱える小さなチャンクに分割し、ベクトル・データベースに格納する
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# そのチャンクをベクターデータベースに埋め込んで保存する
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# 最後に、初期化したvectorstoreからretrieverを作ってみよう
# kは取得するチャンクの数
retriever = vectorstore.as_retriever(k=4)


docs = retriever.invoke("tell me more about that!")


print(docs)
print("-------------------------------------------------------")


query_transform_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
        ),
    ]
)

query_transforming_retriever_chain = RunnableBranch(
    (
        lambda x: len(x.get("messages", [])) == 1,
        # メッセージが1つだけなら、そのメッセージの内容をレトリーバーに渡すだけだ。
        (lambda x: x["messages"][-1].content) | retriever,
    ),
    # もしメッセージがあれば、LLMチェーンに入力を渡してクエリーを変換し、レトリーバーに渡す。
    query_transform_prompt | chat | StrOutputParser() | retriever,
).with_config(run_name="chat_retriever_chain")


# プロンプト テンプレートを定義
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ドキュメントのリストをモデルに渡すためのチェーンを作成します。
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

conversational_retrieval_chain = RunnablePassthrough.assign(
    context=query_transforming_retriever_chain,
).assign(
    answer=document_chain,
)

# ChatMessageHistoryクラスはチャットメッセージの保存と読み込みを行います
demo_ephemeral_chat_history = ChatMessageHistory()


# user側のチャットを保存
demo_ephemeral_chat_history.add_user_message("how can langsmith help with testing?")

response = conversational_retrieval_chain.invoke(
    {"messages": demo_ephemeral_chat_history.messages},
)


demo_ephemeral_chat_history.add_ai_message(response["answer"])

print(response)
print("------------------------------------------------------")
demo_ephemeral_chat_history.add_user_message("tell me more about that!")

response = conversational_retrieval_chain.invoke(
    {"messages": demo_ephemeral_chat_history.messages}
)

print(response)
