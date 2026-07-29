"""
conftest.py — make the project root importable from the tests/ subdirectory,
and patch Neo4j / LLM constructors so that ``import app`` never attempts a
live Neo4j connection during the test session.
"""
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to sys.path so that `import app` works from tests/
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Pre-patch Neo4j and LLM before app.py is imported for the first time.
#
# app.py executes these three statements at module level:
#
#   graph = Neo4jGraph(url=..., username=..., password=...)
#   llm   = get_llm()
#   chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, ...)
#
# Without patches, importing app blocks on the Bolt TCP connection until it
# times out.  We start permanent mock patches here so that every subsequent
# ``import app`` (including the module-level ``from app import …`` at the top
# of test_app.py) sees safe stubs.
# ---------------------------------------------------------------------------

_neo4j_patcher = patch(
    "langchain_neo4j.Neo4jGraph",
    return_value=MagicMock(name="neo4j_graph_instance"),
)
_chain_patcher = patch(
    "langchain_neo4j.GraphCypherQAChain.from_llm",
    return_value=MagicMock(name="graph_cypher_chain_instance"),
)
# get_llm is decorated with @st.cache_resource which requires a live Streamlit
# session context.  Patch the underlying Google GenAI constructor so neither
# the API key validation nor the Streamlit session machinery is triggered.
_llm_patcher = patch(
    "langchain_google_genai.ChatGoogleGenerativeAI",
    return_value=MagicMock(name="chat_google_genai_instance"),
)

_neo4j_patcher.start()
_chain_patcher.start()
_llm_patcher.start()
