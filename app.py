import os
from dotenv import load_dotenv
from uuid import uuid4
from langgraph.prebuilt import create_react_agent
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from pinecone import Pinecone
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage
from langchain_postgres import PGVector
from langmem import create_manage_memory_tool, create_search_memory_tool


from src.prompt import REWRITE_PROMPT, GENERATE_PROMPT




llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    openai_api_type="azure",
)




def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}





def retrive_tool(state: MessagesState):
    """Retrieve data based on user question """
    question = state["messages"][-1].content
    pc = Pinecone(api_key= os.environ.get("PINECONE_API_KEY"))
    index_name = "chatboat-test"
    index = pc.Index(index_name)
    
    docsearch = PineconeVectorStore.from_existing_index(
        index_name = index_name,
        embedding= embeddings_model

    )
    retriver = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    docs = retriver.invoke(question)
    context =  '\n'.join([str(doc.page_content) for doc in docs])
    return {
        "messages": [
            SystemMessage(content=context)
        ]
    }



def answer_generate_tool(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}




from langgraph.config import get_config, get_store


connection = "Connect your databse"
collection_name = "messages"

store = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
namespace = ("agent_memories",)



# Example prompt function
async def prompt(state: AgentState) -> list[AnyMessage]:
    """
    Creates a system prompt for the AI agent using both procedural guidelines
    and contextually relevant memories from the memory store.
    """
    
    config = get_config()
    store = get_store()
    
    
    query_text = state["messages"][-1].content
    
    items = store.search(
        namespace,
        config["configurable"]["user_id"],
        query=query_text
    )
    memories = "\n\n".join([str(item).strip() for item in items])
    
    # Build the system prompt
    system_prompt = f"""
You are a helpful AI assistant. Follow these steps carefully when a user asks a question:

1. Use the `rewrite_question` tool to refine or clarify the original question.
2. Use the `retrive_tool` tool to search and retrieve the most relevant context or documents.
3. Use the `answer_generate_tool` tool to generate a clear and helpful answer
   based on the question and retrieved context.

Use the memories provided below to answer questions contextually:
<memories>
{memories}
</memories>

Guidelines:
- Prioritize information from memories when relevant.
- If the question is unrelated to memories, answer normally.
- Keep explanations concise, accurate, and user-friendly.
"""
    
    # Create system message
    system_message = {"role": "system", "content": system_prompt}
    
    # Return system message + conversation history
    return [system_message, *state["messages"]]




save_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}"),
    actions_permitted=("create", "update"),
    instructions="Update the existing user information...",
    name = "manage_memory",
)
search_tool = create_search_memory_tool(
    namespace=("agent_memories", "{user_id}"),
    instructions="You are a memory search tool. ...",
    store = store
)

agent = create_react_agent(
    model = llm,
    tools = [rewrite_question, retrive_tool, answer_generate_tool, save_memory_tool, search_tool],
    prompt = prompt,  
    store=store,
)

