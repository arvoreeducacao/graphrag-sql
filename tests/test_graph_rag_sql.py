import pytest
from unittest.mock import patch, MagicMock
from main import GraphRagSQL
import os

@pytest.fixture(autouse=True)
def mock_env():
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'}):
        yield

@pytest.fixture
def mock_openai():
    mock_client = MagicMock()
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1] * 10
    mock_response = MagicMock()
    mock_response.data = [mock_embedding_data]
    mock_client.embeddings.create.return_value = mock_response
    return mock_client

@pytest.fixture
def sample_ddl():
    return """
    CREATE TABLE users (
        id INT PRIMARY KEY,
        name VARCHAR(255)
    );

    CREATE TABLE posts (
        id INT PRIMARY KEY,
        user_id INT,
        title VARCHAR(255),
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """

@pytest.fixture
def graph_rag(sample_ddl):
    return GraphRagSQL(sample_ddl, log_level="ERROR", skip_embeddings=True)

def test_graph_creation(graph_rag):
    assert graph_rag.graph is not None
    assert len(graph_rag.graph.nodes) == 7  # 2 tabelas + 5 colunas
    # Verificar tabelas
    assert "users" in graph_rag.graph.nodes
    assert "posts" in graph_rag.graph.nodes
    # Verificar colunas
    assert "users.id" in graph_rag.graph.nodes
    assert "users.name" in graph_rag.graph.nodes
    assert "posts.id" in graph_rag.graph.nodes
    assert "posts.user_id" in graph_rag.graph.nodes
    assert "posts.title" in graph_rag.graph.nodes

def test_graph_edges(graph_rag):
    # Testar todas as arestas (tabela -> coluna)
    table_column_edges = [(s, t) for s, t, d in graph_rag.graph.edges(data=True) 
                         if d.get('type') != 'foreign_key']
    assert len(table_column_edges) == 5  # Cada coluna deve ter uma aresta da sua tabela

def test_foreign_key_relationship(graph_rag):
    # Testar apenas as arestas de chave estrangeira
    fk_edges = [(s, t) for s, t, d in graph_rag.graph.edges(data=True) 
                if d.get('type') == 'foreign_key']
    assert len(fk_edges) == 1
    assert ("posts.user_id", "users.id") in fk_edges

def test_embeddings_skipped(graph_rag):
    assert isinstance(graph_rag.embeddings, dict)
    assert len(graph_rag.embeddings) == 0  # Embeddings devem estar vazios quando skip_embeddings=True 