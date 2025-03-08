import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated

# ✅ Initialize Google Gemini API
genai.configure(api_key="AIzaSyCCEUdNbJcvaT76FTjoWdqY1q3eWwRtQO8")

def query_gemini(role: str, prompt: str) -> str:
    """Query Google Gemini API."""
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(f"You are a {role}. {prompt}")
    return response.text if response else "No response received."

# ✅ Define Workflow State Schema
class WorkflowState(TypedDict):
    feature_request: Annotated[str, "single"]
    product_vision: Annotated[str, "single"]
    retrieved_knowledge: Annotated[str, "single"]
    backlog_priorities: Annotated[str, "single"]
    technical_feasibility: Annotated[str, "single"]
    ux_design: Annotated[str, "single"]
    execution_plan: Annotated[str, "single"]
    retrospective_analysis: Annotated[str, "single"]
    okr_insights: Annotated[str, "single"]

# ✅ Load and Vectorize Knowledge Base
loader = TextLoader("agile_knowledge_base.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(texts, embeddings)

# ✅ Define Agile Workflow Functions
def product_manager(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are a **Product Manager** following **SVPG Agile** principles.
    - Define the **product vision** for the feature request: "{state['feature_request']}".
    - Identify the **customer problem** it solves.
    - Prioritize based on **market demand, business goals, and usability**.
    """
    return {"product_vision": query_gemini("Product Manager", prompt)}

def rag_retrieval_agent(state: WorkflowState) -> WorkflowState:
    docs = vector_store.similarity_search(state["feature_request"], k=2)
    return {"retrieved_knowledge": "\n".join([doc.page_content for doc in docs])}

def safe_product_owner(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are a **SAFe Product Owner** managing an Agile Release Train (ART).
    - Translate the product vision into an **Agile backlog**.
    - Prioritize tasks based on **retrieved insights** and SAFe PI Planning principles.
    - Ensure business alignment.
    """
    return {"backlog_priorities": query_gemini("SAFe Product Owner", prompt)}

def tech_lead(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are a **Tech Lead** following **SVPG Agile principles**.
    - Assess **technical feasibility** of backlog priorities.
    - Recommend **architecture, scalability, and tech stack**.
    """
    return {"technical_feasibility": query_gemini("Tech Lead", prompt)}

def ux_designer(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are a **UX Designer** focused on usability.
    - Define key **user personas** for the product.
    - Propose **UI/UX improvements** for the best customer experience.
    """
    return {"ux_design": query_gemini("UX Designer", prompt)}

def delivery_manager(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are a **SAFe Agile Delivery Manager**.
    - Identify dependencies and risks in execution.
    - Recommend a **release plan** aligned with **Program Increment (PI) Planning**.
    """
    return {"execution_plan": query_gemini("Delivery Manager", prompt)}

def sprint_retrospective(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are an AI Agile Coach conducting a **Sprint Retrospective**.
    - Identify successes and failures in execution.
    - Suggest actionable improvements for the next sprint.
    """
    return {"retrospective_analysis": query_gemini("Agile Sprint Coach", prompt)}

def okr_tracking(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are an AI system tracking **Agile OKRs**.
    - Measure progress against **business objectives**.
    - Identify key success metrics and gaps.
    """
    return {"okr_insights": query_gemini("OKR Tracker", prompt)}

# ✅ Create LangGraph Workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("product_manager", product_manager)
workflow.add_node("rag_retrieval_agent", rag_retrieval_agent)
workflow.add_node("safe_product_owner", safe_product_owner)
workflow.add_node("tech_lead", tech_lead)
workflow.add_node("ux_designer", ux_designer)
workflow.add_node("delivery_manager", delivery_manager)
workflow.add_node("sprint_retrospective", sprint_retrospective)
workflow.add_node("okr_tracking", okr_tracking)
workflow.add_node("end", lambda state: state)  # Explicit End Node

# ✅ Define Workflow Structure
workflow.set_entry_point("product_manager")
workflow.add_edge("product_manager", "rag_retrieval_agent")
workflow.add_edge("rag_retrieval_agent", "safe_product_owner")
workflow.add_edge("safe_product_owner", "tech_lead")
workflow.add_edge("safe_product_owner", "ux_designer")
workflow.add_edge("tech_lead", "delivery_manager")
workflow.add_edge("delivery_manager", "sprint_retrospective")
workflow.add_edge("delivery_manager", "okr_tracking")
workflow.add_edge("sprint_retrospective", "end")
workflow.add_edge("okr_tracking", "end")

# ✅ Streamlit UI
st.title("Agile AI Workflow")
feature_request = st.text_input("Enter Feature Request:", "Build a real-time AI analytics dashboard for enterprise customers.")

if st.button("Run Workflow"):
    graph = workflow.compile()
    initial_state = {"feature_request": feature_request}
    result = graph.invoke(initial_state)
    
    st.subheader("Workflow Results")
    for key, value in result.items():
        st.write(f"**{key.replace('_', ' ').title()}**:")
        st.write(value)
