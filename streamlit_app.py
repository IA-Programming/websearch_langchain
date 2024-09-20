import warnings

warnings.filterwarnings("ignore")
import streamlit as st
import os
from json import load
st.set_page_config(page_title="Interweb Chat Explorer", page_icon="🌐", initial_sidebar_state="auto")
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"

from pathlib import Path
from typing import Literal, Dict
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
# memory and prompt
from langchain.memory import ConversationBufferMemory
# Streamlit UI Callback
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# Retrieval Chain
# from langchain.chains.question_answering.map_rerank_prompt import PROMPT as MAP_RERANK_PROMPT
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
# Vectorstore
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores.docarray.base import DocArrayIndex
from docarray.index import InMemoryExactNNIndex
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai import ChatVertexAI
# Search
# from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_google_community.search import GoogleSearchAPIWrapper
# Retriver used
from langchain.chains.llm import LLMChain
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers.web_research import WebResearchRetriever, QuestionListOutputParser
# from langchain.chains.qa_with_sources.map_reduce_prompt import EXAMPLE_PROMPT
# Custom Modules
from InterwebPrompts import EXAMPLE_PROMPT, QUESTION_PROMPT, COMBINE_CHAT_PROMPT, Search_ChatPrompt, DEFAULT_REFINE_PROMPT, DEFAULT_TEXT_QA_PROMPT, MAP_RERANK_PROMPT, get_current_date
from utils import set_vertex_ai_credentials, delete_session, add_text, about_us
from capturing_callback_handler import playback_callbacks

if 'sources' not in st.session_state:
    st.session_state['sources'] = []
if 'session_started' not in st.session_state:
    st.session_state.session_started = False
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'memory' not in st.session_state and 'search_memory' not in st.session_state:
    smjs = StreamlitChatMessageHistory(key="lang_key")
    st.session_state['search_memory'] = ConversationBufferMemory(chat_memory=smjs, input_key='question', return_messages=True, memory_key="memory", output_key="text")
    msgs = StreamlitChatMessageHistory(key="mensajes")
    st.session_state['memory'] = ConversationBufferMemory(chat_memory=msgs, input_key='question', return_messages=True, memory_key="chat_history", output_key="answer")

SAVED_SESSIONS = ["Upcoming tech conferences", "When is Mother's Day?", "The most awaited movies for 2025", "How many weeks are in a year?"]

def settings(model_name:str, tipo:Literal["stuff", "map_reduce", "refine", "map_rerank"], settings: Dict):
    # Generate answer (w/ citations)
    import logging
    logging.basicConfig()
    logging.getLogger("langchain_community.retrievers.WebResearchRetriever").setLevel(logging.INFO)

    model = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, cache_folder="./"
    )
    metric: Literal["cosine_sim", "euclidian_dist", "sgeuclidean_dist"] = "cosine_sim"
    doc_cls = DocArrayIndex._get_doc_cls(space=metric)
    doc_index = InMemoryExactNNIndex[doc_cls]()
    vectorstore_public = DocArrayInMemorySearch(doc_index=doc_index, embedding=embeddings_model)

    llm: BaseChatModel = ChatVertexAI(model=model_name, temperature=0, streaming=True, location="us-east5", max_output_tokens=4096)

    search: GoogleSearchAPIWrapper = GoogleSearchAPIWrapper(k=1)   

    memoria: ConversationBufferMemory = st.session_state.get('search_memory')

    DEFAULT_SEARCH_PROMPT = Search_ChatPrompt.partial(date=get_current_date())

    cadena = LLMChain(prompt=DEFAULT_SEARCH_PROMPT, llm=llm, output_parser=QuestionListOutputParser(), memory=memoria, verbose=True)
    
    web_retriever: BaseRetriever = WebResearchRetriever(
        vectorstore=vectorstore_public,
        llm_chain=cadena, 
        search=search, 
        num_search_results=1,
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=150),
        # trust_env=True,
        allow_dangerous_requests=True,
    )

    memory: ConversationBufferMemory = st.session_state.get('memory')

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,
                                                        retriever=web_retriever,
                                                        chain_type=tipo, 
                                                        return_source_documents=True, 
                                                        memory=memory, 
                                                        chain_type_kwargs=settings,
                                                            )
    return qa_chain

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

st.header("`Interweb Chat Explorer`")
st.info("`I am an AI that can answer questions by exploring, reading, and summarizing web pages."
    "I can be configured to use different modes: public API or private (no data sharing).`")
with st.sidebar:
    st.title("Interweb Chat Explorer")
    with st.expander(label="Settings", expanded=(not st.session_state.get('session_started', False))):
        if st.session_state.get('mensaje', False):
            if 'exist'in st.session_state.mensaje: st.warning(body=st.session_state.mensaje, icon="⚠️")
            else: st.success(body=st.session_state.mensaje, icon="✅")
            del st.session_state.mensaje
        chain_kwargs: Dict = {}
        if not st.session_state.session_started and 'ruta_saved' not in st.session_state:
            set_vertex_ai_credentials()
        else:
            model_name: str = st.selectbox(label="🔌 Select Model", options=("gemini-1.0-pro-002", "gemini-1.5-flash-001", "gemini-1.5-pro-preview-0409"),index=0 ,help="Here you can select the model you want to use")

            option = st.selectbox(
            "Select the type of QA-Chain",
            ("stuff", "map_reduce", "refine", "map_rerank"),
            help="type LLM summarization chain"
            )
            delete_session()
            fecha: str = get_current_date()
            chain_kwargs.update({"document_prompt": EXAMPLE_PROMPT, "verbose": True})
            if option=='stuff':
                COMBINE_CHAT_PROMPT_PARTIAL = COMBINE_CHAT_PROMPT.partial(date=fecha)
                chain_kwargs.update({"prompt": COMBINE_CHAT_PROMPT_PARTIAL, "document_variable_name": "summaries"})
            elif option=='map_reduce':
                QUESTION_PROMPT_PARTIAL = QUESTION_PROMPT.partial(date=fecha)
                COMBINE_CHAT_PROMPT_PARTIAL = COMBINE_CHAT_PROMPT.partial(date=fecha)
                chain_kwargs.update({"question_prompt": QUESTION_PROMPT_PARTIAL, "combine_prompt": COMBINE_CHAT_PROMPT_PARTIAL, "combine_document_variable_name": "summaries", "map_reduce_document_variable_name": "context", "token_max": 15000})
            elif option=='refine':
                DEFAULT_TEXT_QA_PROMPT_PARTIAL = DEFAULT_TEXT_QA_PROMPT.partial(date=fecha)
                DEFAULT_REFINE_PROMPT_PARTIAL = DEFAULT_REFINE_PROMPT.partial(date=fecha)
                chain_kwargs.update({"question_prompt": DEFAULT_TEXT_QA_PROMPT_PARTIAL, "refine_prompt": DEFAULT_REFINE_PROMPT_PARTIAL, "document_variable_name": "context_str", "initial_response_name": "existing_answer"})
            elif option=='map_rerank':
                chain_kwargs.update({"prompt": MAP_RERANK_PROMPT, "document_variable_name": "context", "rank_key": "score", "answer_key": "answer"})
                chain_kwargs.pop("document_prompt")

    if st.button(label="Clear Chat History", help="when you want to start a new chat",disabled=False if st.session_state.form_submitted else True):
        st.session_state.get('memory').clear()
        st.session_state.get('search_memory').clear()
        st.session_state.form_submitted = False
        st.rerun()

    about_us()

# Make retriever and llm
if len(chain_kwargs)!=0:
    if 'qa_chain' not in st.session_state or chain_kwargs!=st.session_state.get("session_kwargs"):
        st.session_state['qa_chain'] = settings(model_name=model_name, tipo=option, settings=chain_kwargs)
        st.session_state["session_kwargs"] = chain_kwargs
# User input
if not st.session_state.form_submitted:
    with st.form("search_form"):
        if st.session_state.session_started:
            user_input = st.text_input("Enter your search query", help="Here you can put all the doubts that you heart have")    
        else:
            user_input = st.selectbox(label="Here you have some test questions", options=SAVED_SESSIONS, index=0, help="Select some testing questions to see with what are we working")
        submitted = st.form_submit_button(label="Search")
        if submitted and user_input:
            st.session_state.form_submitted = True
            st.session_state.question = user_input
            st.rerun()
# To render your history messages
if len(st.session_state.get('lang_key'))!=0 and len(st.session_state.get('mensajes'))!=0:
    for index, (search_msj, hist_message) in enumerate(zip(st.session_state.get('lang_key'), st.session_state.get('mensajes'))):
        with st.chat_message("assistant" if hist_message.type=="ai" else "user"):
            respuesta = "## `Answer:`\n\n" if hist_message.type=="ai" else ""
            st.write(respuesta + hist_message.content)
            if hist_message.type=="ai":
                with st.status(label="**Sources:**", state="complete"):
                    st.markdown(body=str("\n---\n".join([f"""### Webpage {i+1}.-\n\n{doc}""" for i, doc in enumerate(st.session_state.sources[int(index//2)])])))
if st.session_state.form_submitted:
    user_input = st.chat_input("Here goes your question", disabled=True if not st.session_state.session_started else False)

    if user_input or st.session_state.get("question", False):
        
        qa_chain: BaseQAWithSourcesChain = st.session_state.get("qa_chain")
        question = user_input or st.session_state.get("question")
        st.chat_message(name="user").markdown(question)
        if question in SAVED_SESSIONS:
            with st.chat_message(name="assistant", avatar="🦜"):
                with st.spinner("Generating your Answer"):
                    answer = st.empty()
                    stream_handler = StreamHandler(answer, initial_text="## `Answer:`\n\n")
                    nombre = add_text(question)
                    Ruta = (Path(__file__).parent / f"saves/{nombre}.pickle").absolute()
                    result = playback_callbacks(handlers=[stream_handler], records_or_filename=Ruta, max_pause_time=1)
                    answer.markdown('## `Answer:`\n\n' + result['answer'])
                    unique_items = list(set([f"""[{doc.metadata['title']}]({doc.metadata['source']})""" for doc in result['source_documents']]))
                    st.session_state.sources.append(unique_items)
                    with st.status(label="**Sources:**",state="complete"):
                        st.markdown(str("\n---\n".join([f"""### Webpage {i+1}.-\n\n{doc}""" for i, doc in enumerate(unique_items)])))
                    search_memory: ConversationBufferMemory = st.session_state.search_memory
                    memory: ConversationBufferMemory = st.session_state.memory
                    data = load(open((Path(__file__).parent / "saves/search_memory.json").absolute()))
                    search_memory.save_context(inputs={"question": question}, outputs={"text": data[SAVED_SESSIONS.index(question)]["answer"]})
                    memory.save_context(inputs={"question": question}, outputs={"answer": result["answer"]})
        else:
            # Write answer and sources
            with st.chat_message(name="assistant", avatar="🦜"):
                with st.spinner("Generating your Answer"):
                    answer = st.empty()
                    stream_handler = StreamHandler(answer, initial_text="## `Answer:`\n\n")
                    result = qa_chain.invoke(
                        input={"question": question}, 
                        return_only_outputs=True, 
                        config={"callbacks":[
                                            stream_handler,
                                            ]}
                        )
                    answer.markdown('## `Answer:`\n\n' + result['answer'])
                    unique_items = list(set([f"""[{doc.metadata['title']}]({doc.metadata['source']})""" for doc in result['source_documents']]))
                    st.session_state.sources.append(unique_items)
                    with st.status(label="**Sources:**",state="complete"):
                        st.markdown(str("\n---\n".join([f"""### Webpage {i+1}.-\n\n{doc}""" for i, doc in enumerate(unique_items)])))
        if st.session_state.get("question", False):
            del st.session_state.question