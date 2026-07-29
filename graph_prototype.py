import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graph_transformers.llm import LLMGraphTransformer

def main():
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    # 1. Connect to the Neo4j Database
    print("Connecting to Neo4j...")
    graph = Neo4jGraph(
        url="bolt://localhost:7687", 
        username="neo4j", 
        password="password"
    )

    # 2. Initialize LLM and Transformer
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    allowed_nodes = ["Concept", "Component", "Process", "Database_Object"]
    
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes
    )

    # 3. Sample Data
    sample_text = """
    A Relational Database consists of multiple Tables. 
    Tables contain Primary Keys. 
    A Primary Key uniquely identifies a Record.
    Foreign Keys establish relationships between Tables.
    Boyce-Codd Normal Form eliminates redundancy in Tables.
    """
    documents = [Document(page_content=sample_text)]

    # 4. Extract and Store
    print("Extracting graph topology...")
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    if graph_documents:
        print("Storing data in Neo4j...")
        graph.add_graph_documents(graph_documents)
        print("Success! The graph is now live in Neo4j.")
    else:
        print("No graph data extracted.")

if __name__ == "__main__":
    main()