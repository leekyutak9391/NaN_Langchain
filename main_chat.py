import streamlit as st
import tiktoken
from loguru import logger
import os

from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI

# from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import Docx2txtLoader
# from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings

# from langchain.memory import ConversationBufferMemory
# from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from PIL import Image
from langchain_openai.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI


from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import OpenAIEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.memory import StreamlitChatMessageHistory

#DB관련
#import cx_Oracle
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain.agents.agent_types import AgentType
from langchain.agents import create_sql_agent
#import oracledb

def main():
    st.set_page_config(
    page_title="운영리스크",
    #page_icon=":books:"
    )
    OPENAI_API_KEY="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"
    #os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    #LOCATION = "D:\instantclient_18_5"
    #os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]

    # lib_dirs = os.path.join('D:\\app\\nan\\product\\21c\\homes\\OraDB21Home1\\network', 'admin')
    #cx_Oracle.init_oracle_client(lib_dir="D:\instantclient_18_5")
    #DB 접속정보 확인 로직 (오라클)
    # hostname='DESKTOP-5DMH5VA'
    # port='1521'
    # service_name='xe'
    # username='<your db="npl_test" user="system">'
    # password='<your password="diafkfl111">'
    
    #cx_Oracle.init_oracle_client(lib_dir=lib_dirs)
    #cx_Oracle.init_oracle_client(lib_dir="D:\app\nan\product\21c\homes\OraDB21Home1\network\admin")
    # oracle_connection_string_fmt = (
    #     'oracle+cx_oracle://{username}:"{password}"@' +
    #     cx_Oracle.makedsn('{hostname}', '{port}', service_name='{service_name}')
    # )
    # engine = create_engine(
    #         oracle_connection_string_fmt.format(
    #         username=username, password=password, 
    #         hostname=hostname, port=port, 
    #         service_name=service_name,
    #     )
    # )

    #st.write(engine)
    #단순 db 접속(확인 완료)
    #oracledb.init_oracle_client(lib_dir="D:\instantclient-basic-windows.x64-21.12.0.0.0dbru\instantclient_21_12")
    #oracle_con = oracledb.connect(user="system", password="diafkfl22@", dsn="localhost:1521/XE")
    
    
    st.title("운영리스크 :blue[(NaN)] :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        options = st.selectbox(
        '원하는 질문의 카테고리를 선택해 주세요.',
        ['전체','운영리스크','측정', '손실', 'BCP','DB조회']
        )
        uploaded_files =  st.file_uploader(" * 운영리스크 파일 업로드 (상단의 카테고리 설정 후 저장해주세요.)",type=['pdf','docx','xlsx'],accept_multiple_files=True)
        #openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("문서 학습")
        #connect_db = st.button("DB 연동")
    if process:
        if len(uploaded_files) == 0:
            st.markdown('''
            ⚠️업로드할 문서를 지정해주세요.
            ''')
            #st.warning('업로드할 문서를 지정해주세요.', icon="⚠️")
        else:
            #if not openai_api_key:
            #    st.info("Please add your OpenAI API key to continue.")
            #    st.stop()
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vetorestore = get_vectorstore(text_chunks,options)

            st.session_state.conversation = get_conversation_chain(vetorestore,OPENAI_API_KEY) 
            st.markdown('''
            문서 업로드가 완료되었습니다.
            ''')
            st.session_state.processComplete = True
        
    # if st.session_state.processComplete == True:
    #     uploaded_files = []
    #if connect_db:
    #    db=SQLDatabase.from_uri("system@//localhost:1521/xe")
        
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 운영리스크에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            i = 0
            #st.write(chain)
            with st.spinner("Thinking..."):
                if len(uploaded_files) != 0:
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']
                    #logger.info(source_documents)
                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        while i < len(source_documents):
                            st.markdown(source_documents[i].metadata['source'], help = source_documents[i].page_content)
                            #st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                            #st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                            if i == 2: #참고 문서 확인칸 제한
                                break
                            i = i+1
                else:
                    embeddings = get_embeddings()
                    convert_options = convert_option(options)
                    #save_url = "/faiss_db/orms/"+convert_options
                    save_db_url = "https://raw.githubusercontent.com/leekyutak9391/NaN_Langchain/main/faiss/"
                    if convert_options == "los_data":
                        try:
                            #llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name = 'gpt-3.5-turbo',temperature=0)
                            llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',temperature=0)
                            #engine=create_engine(url, echo=True)
                        
                            #db=SQLDatabase.from_uri('oracle://langchain:nan1234@DESKTOP-5DMH5VA:1521/xe')  #로컬DB
                            db=SQLDatabase.from_uri('oracle://JBFG_ORMS:JBFG_ORMS@222.236.44.99:1521/ORCL')  # 44.99 DB
                            #st.write(db)
                        
                            custom_suffix = """
                            Information about the database cannot be viewed.
                            Information about the table cannot be viewed.
                            Information about the column cannot be viewed.
                            The physical name of the table cannot be searched.
                            Column names in the table cannot be searched.
                            First, I need to get some similar examples that I know of.
                            If the examples are sufficient to construct your query, you can write your query.
                            Otherwise, you can look at the tables in your database to see what you can query.
                            You then need to query the schema of the most relevant tables.
                            """

                            agent_executor = create_sql_agent(
                                llm=llm,
                                toolkit=SQLDatabaseToolkit(db=db, llm=llm),
                                verbose=True,
                                agent_type=AgentType.OPENAI_FUNCTIONS, #자연어 처리관련
                                suffix = custom_suffix,
                                handle_parsing_errors=True
                            )
                        
                            #chain = SQLDatabaseChain.from_llm(llm,db)
                            response = agent_executor.run(query)
                            st.markdown(response)
                        except:
                            response = "조회가 불가능한 질문입니다."
                            st.markdown(response)
                    else:
                        try:
                            vetorestore = FAISS.load_local(save_db_url,embeddings)
                            similarity_search = vetorestore.similarity_search_with_score(query) # 유사도 점수 출력 가능.
                            chain = get_conversation_chain(vetorestore,OPENAI_API_KEY)
                            result = chain({"question": query})
                            #st.write(vetorestore)
                            #result_content = new_db.similarity_search_with_score(content,3)
                            response = result['answer']
                            source_documents = result['source_documents']
                            #st.markdown(similarity_search)
                            #st.write(result)
                            st.markdown(response)
                            #if similarity_search[]
                            with st.expander("참고 문서 확인"):
                                while i < len(source_documents):
                                    st.markdown(source_documents[i].metadata['source'], help = source_documents[i].page_content)
                                    if i == 2: #참고 문서 확인칸 제한
                                        break
                                    i = i+1
                        except:
                            response = "조회가 불가능한 질문입니다."
                            st.markdown(response)
                        #st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                        #st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                        #st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            #logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        elif '.xlsx' in doc.name:
            loader = UnstructuredExcelLoader(file_name)
            documents = loader.load_and_split()
            
        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )
    return embeddings

def get_vectorstore(text_chunks,options):
    embeddings = get_embeddings()
    convert_options = convert_option(options)
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    #save_db_url = "/faiss_db/orms/"+convert_options
    save_db_url = "https://raw.githubusercontent.com/leekyutak9391/NaN_Langchain/main/faiss/"
    #st.write(save_db_url)
    vectordb.save_local(save_db_url)   #로컬에 저장할 수 있음. 참조 : https://www.youtube.com/watch?v=fE4SH2vEsdk&t=1061s
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    #llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2}, vervose = True), #체인 과정에서 유사도 관련 설정  score_threshold 점수가 낮을 수록 유사도가 높음.
            #retriever=vetorestore.as_retriever(search_type="mmr", vervose = True), #최대 한계 유사도
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )
    
    return conversation_chain

def convert_option(options):
    if options == "전체":
        convert_options = ""
    elif options == "운영리스크":
        convert_options = "orms"
    elif options == "측정":
        convert_options = "msr"
    elif options == "손실":
        convert_options = "los"
    elif options == "BCP":
        convert_options = "bcp"
    elif options == "DB조회":
        convert_options = "los_data"
    return convert_options

#def create_engine_ora(self):
    
    
if __name__ == '__main__':
    main()
