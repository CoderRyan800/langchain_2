#### CODE BELOW SETS UP THE BASIC AGENT ####

import bs4
import traceback
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
# import asyncio
# import aiosqlite
# from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

import sqlite3
con = sqlite3.connect("test_agent_1.db",check_same_thread=False)

memory = SqliteSaver(con)
llm = ChatOpenAI(model="gpt-4o", temperature=0)


### Construct retriever ###
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

conversation_history_filename = 'Conversation_History.txt'

try:
    fp = open(conversation_history_filename,'r')
    conversation_data = fp.read()
    fp.close()
except:
    fp = open(conversation_history_filename,'w')
    fp.write("BEGIN CONVERSATION HISTORY\n")
    fp.close()
    fp = open(conversation_history_filename,'r')
    conversation_data = fp.read()
    fp.close()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
#splits = text_splitter.create_documents([conversation_data])
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())




retriever = vectorstore.as_retriever()


### Build retriever tool ###
tool = create_retriever_tool(
    retriever,
    "data_retriever",
    "Searches and returns long-term information",
)
tools = [tool]

agent_executor = create_react_agent(llm, tools, checkpointer=memory)

#### END CODE TO DEFINE THE CORE AGENT

#### CONFIGURATION IS USED TO SET UP CONVERSATION THREADS USING A THREAD_ID ####

config = {"configurable": {"thread_id": "abc123"}}

#### END CONVERSATION THREAD CONFIGURATION ####

flag_stop_agent = False

fp = open(conversation_history_filename,'a')

try:
    while not flag_stop_agent:

        input_string_from_user = input("Enter message for the agent.  Type \"EXIT\" to exit: ")

        if input_string_from_user == "EXIT":
            flag_stop_agent = True 
            break

        for s in agent_executor.stream({"messages": [HumanMessage(content=input_string_from_user)]}, config=config):
            agent_message = s['agent']['messages'][0].content
            print(s['agent']['messages'][0].content)
            fp.write("HUMAN: %s\n" % (input_string_from_user,))
            fp.write("AGENT: %s\n" % (agent_message,))
            print("----")
except Exception as e:
    traceback.print_exception(e)
    traceback.print_exc()
finally:
    # VERY CRITICAL: CLOSE THE DATABASE!
    try:
        con.close()
        fp.close()
    except:
        print("ERROR: UNABLE TO CLOSE THE SQLITE DATABASE FOR THE AGENT!")


# End the while loop

