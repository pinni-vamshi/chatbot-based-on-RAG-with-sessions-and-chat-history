import bs4
import os
import getpass
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#getting acess to groq model
os.environ["GROQ_API_KEY"] = getpass.getpass() #this will ask for a API key
llm = ChatGroq(model = "llama3-8b-8192")

#getting the content from the web page and parsing it to get post-content, post title , post header
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
)
docs = loader.load()

#splitting the docs into multiple chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
splits = text_splitter.split_documents(docs)
### print(splits)

#converting those splits into vectors using hyggingface_embedding model
vectorstore = SKLearnVectorStore.from_documents(documents=splits , embedding=HuggingFaceEmbeddings())

#retrieve those splits which are similar to the input
retriever = vectorstore.as_retriever() #uses similarity search by default


######## WE NEED TO MODIFY THE QUESTION AND THE RETRIEVER BASED ON THE CHAT HISTORY


#uses chat histroy, input and this prompt to create a contextualized Question prompt
contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder( "chat_history"),
    ("human" , "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm , retriever , contextualize_q_prompt)

#conceptualizing the retriever means the retriever get the relevant embeddings based on the input , chat_history, ( conceptualizing: here llm combines both and create a relevant query for the retriever)
system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

#now we got the modified retriever and question based on the history we can use the model to the answers
rag_chain = create_retrieval_chain(history_aware_retriever , question_answer_chain)

#creating a sessions(different users) that stores chat histories for different sessions
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id  not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history,input_messages_key="input",  history_messages_key="chat_history",
   output_messages_key="answer", )

#example input
response1 = conversational_rag_chain.invoke({"input":"what is task decomposition?"}, config = {"configurable":{"session_id": "abc1"}})
print("response1: ",response1["answer"])
response2 = conversational_rag_chain.invoke({"input":"Can you explain in points?"}, config = {"configurable":{"session_id": "abc1"}})
print("response2: ",response2["answer"])
