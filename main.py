"""
GraphRagSQL: Uma ferramenta para análise de esquemas SQL usando grafos e RAG.

Este módulo implementa a conversão de DDL SQL em grafos de conhecimento,
permitindo análise semântica e consultas baseadas em linguagem natural.
"""

import sqlparse
import os
import networkx as nx
import asyncio
import pickle
from datetime import datetime
from openai import OpenAI
from typing import List, Dict, Set, Tuple, Optional
import json
from scipy.spatial.distance import cosine
import math
import re

class Logger:
    """
    Sistema de logging com suporte a múltiplos níveis de log.
    
    Attributes:
        log_level (str): Nível de log atual (DEBUG, INFO, WARNING, ERROR)
    """
    
    def __init__(self, log_level: str = "INFO") -> None:
        """
        Inicializa o logger com um nível específico.
        
        Args:
            log_level (str): Nível de log desejado
        """
        self.log_level = log_level
        
    def _log(self, message: str, level: str = "INFO") -> None:
        """
        Método interno para logging com timestamp.
        
        Args:
            message (str): Mensagem a ser logada
            level (str): Nível do log
        """
        log_levels = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3
        }

        current_level = log_levels.get(self.log_level.upper(), 1)
        message_level = log_levels.get(level.upper(), 1)

        if message_level >= current_level:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level.upper()}: {message}")

    def info(self, message: str) -> None:
        """Loga uma mensagem no nível INFO."""
        self._log(message, "INFO")

    def debug(self, message: str) -> None:
        """Loga uma mensagem no nível DEBUG."""
        self._log(message, "DEBUG")

    def warning(self, message: str) -> None:
        """Loga uma mensagem no nível WARNING."""
        self._log(message, "WARNING")

    def error(self, message: str) -> None:
        """Loga uma mensagem no nível ERROR."""
        self._log(message, "ERROR")


class GraphRagSQL:
    """
    Classe principal que converte DDL SQL em um grafo de conhecimento.
    
    Esta classe analisa DDL SQL, cria um grafo direcionado representando
    as relações entre tabelas e colunas, e gera embeddings para análise
    semântica.
    
    Attributes:
        ddl (str): O DDL SQL a ser analisado
        logger (Logger): Sistema de logging
        graph (nx.DiGraph): Grafo direcionado representando o esquema
        embeddings (Dict[str, List[float]]): Embeddings das entidades do grafo
    """
    
    def __init__(self, ddl: str, log_level: str = "INFO", skip_embeddings: bool = False) -> None:
        """
        Inicializa o GraphRagSQL com um DDL específico.
        
        Args:
            ddl (str): O DDL SQL a ser analisado
            log_level (str): Nível de log desejado
            skip_embeddings (bool): Se True, pula a geração de embeddings (útil para testes)
        """
        self.ddl: str = ddl
        self.logger = Logger(log_level)
        self.logger.info("Starting GraphRagSQL...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.graph: nx.DiGraph = self._parse_ddl_to_graph(ddl)
        if skip_embeddings:
            self.embeddings = {}
        else:
            self.embeddings: Dict[str, List[float]] = self._load_or_generate_embeddings(self.graph)
        self.logger.info("GraphRagSQL initialized successfully")
        
    def _parse_ddl_to_graph(self, ddl: str) -> nx.DiGraph:
        self.logger.debug("Starting DDL to graph parsing")
        graph = nx.DiGraph()
        parsed = sqlparse.parse(ddl)

        for stmt in parsed:
            if stmt.get_type() == "CREATE":
                for token in stmt.tokens:
                    if isinstance(token, sqlparse.sql.Identifier):
                        table_name = token.get_name()
                        self.logger.debug(f"Processing table: {table_name}")
                        graph.add_node(table_name, type="table")

                        for sub_token in stmt.tokens:
                            if isinstance(sub_token, sqlparse.sql.Parenthesis):
                                content = sub_token.value.strip('()')
                                column_definitions = self._split_column_definitions(content)

                                for column_def in column_definitions:
                                    column_def = column_def.strip()
                                    
                                    if column_def.lower().startswith('foreign key'):
                                        self._process_foreign_key(graph, table_name, column_def)
                                        continue

                                    column_parts = column_def.split(None, 1)
                                    if len(column_parts) >= 1:
                                        column_name = column_parts[0]
                                        if column_name.upper() not in ('CONSTRAINT', 'FOREIGN', 'PRIMARY'):
                                            column_full_name = f"{table_name}.{column_name}"
                                            self.logger.debug(f"Adding column: {column_full_name}")
                                            graph.add_node(column_full_name,
                                                         type="column",
                                                         definition=column_def)
                                            graph.add_edge(table_name, column_full_name)
                        break
        return graph

    def _split_column_definitions(self, content: str) -> List[str]:
        """Divide as definições de colunas respeitando parênteses aninhados."""
        definitions = []
        current_def = []
        paren_count = 0

        for char in content:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                definitions.append(''.join(current_def).strip())
                current_def = []
                continue
            current_def.append(char)

        if current_def:
            definitions.append(''.join(current_def).strip())

        return definitions

    def _process_foreign_key(self, graph: nx.DiGraph, table_name: str, fk_def: str) -> None:
        """Processa uma definição de chave estrangeira e adiciona ao grafo."""
        try:
            # Extrair coluna local
            local_col_match = re.search(r'foreign key\s*\(([^)]+)\)', fk_def.lower())
            if not local_col_match:
                return
            local_col = local_col_match.group(1).strip()
            
            # Extrair tabela e coluna referenciada
            ref_match = re.search(r'references\s+(\w+)\s*\(([^)]+)\)', fk_def.lower())
            if not ref_match:
                return
            ref_table, ref_col = ref_match.group(1).strip(), ref_match.group(2).strip()

            source = f"{table_name}.{local_col}"
            target = f"{ref_table}.{ref_col}"

            self.logger.debug(f"Adding foreign key: {source} -> {target}")
            graph.add_edge(source, target, type="foreign_key")

        except Exception as e:
            self.logger.error(f"Error processing foreign key: {str(e)}")

    async def _get_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )

        return [item.embedding for item in response.data]

    async def _generate_embeddings_async(self, graph: nx.DiGraph) -> Dict[str, List[float]]:
        nodes = list(graph.nodes)
        batch_size = 500
        total_batches = math.ceil(len(nodes) / batch_size)
        embeddings = {}

        self.logger.info(f"Generating embeddings for {len(nodes)} nodes in {total_batches} batches")

        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i:i + batch_size]

            self.logger.debug(f"Generating embeddings for batch {i + 1} of {total_batches}")
            batch_embeddings = await self._get_embedding_batch(batch_nodes)

            for node, embedding in zip(batch_nodes, batch_embeddings):
                embeddings[node] = embedding

        return embeddings

    def _load_or_generate_embeddings(self, graph: nx.DiGraph, embeddings_file: str = 'graph_rag_sql_embeddings.pkl') -> Dict[str, List[float]]:
        self.logger.debug(f"Checking embeddings in: {embeddings_file}")
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                cached_data = pickle.load(f)
                cached_embeddings = cached_data['embeddings']
                cached_nodes = cached_data['nodes']

                current_nodes = set(graph.nodes())
                if current_nodes == set(cached_nodes):
                    self.logger.info("Using cached embeddings")
                    return cached_embeddings

        self.logger.info("Generating new embeddings...")
        embeddings = asyncio.run(self._generate_embeddings_async(graph))

        with open(embeddings_file, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'nodes': list(graph.nodes()),
                'timestamp': datetime.now()
            }, f)
        self.logger.info("New embeddings generated and saved successfully")
        return embeddings

    def export_graph_json(self, output_file: str = 'graph_data.json') -> None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        graph_data = {
            "nodes": [],
            "links": []
        }

        for i, (node, attr) in enumerate(self.graph.nodes(data=True)):
            if i < 5:
                pass

        for i, (source, target, attr) in enumerate(self.graph.edges(data=True)):
            if i < 5:
                pass

        for node, attr in self.graph.nodes(data=True):
            node_data = {
                "id": node,
                "name": node.split('.')[-1] if '.' in node else node,
                "val": 30 if attr.get('type') == 'table' else 20,
                "group": "table" if attr.get('type') == 'table' else "column",
                "definition": attr.get('definition', '')
            }
            graph_data["nodes"].append(node_data)

        for source, target, attr in self.graph.edges(data=True):
            link_data = {
                "source": source,
                "target": target,
                "type": attr.get('type', 'table_column')
            }
            graph_data["links"].append(link_data)

        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)

    async def _query_graph_async(self, query: str) -> List[Tuple[str, float]]:
        self.logger.debug(f"Executing query: {query}")
        query_embedding = await self._get_embedding_batch([query])
        similarities = {
            node: 1 - cosine(query_embedding[0], emb)
            for node, emb in self.embeddings.items()
        }
        sorted_nodes = sorted(similarities.items(),
                              key=lambda x: x[1], reverse=True)
        top_matches = sorted_nodes[:3]
        self.logger.debug(f"Top 3 matches found: {[match[0] for match in top_matches]}")
        return top_matches

    def _get_related_tables(self, node_name: str) -> Set[str]:
        related_tables = set()

        table_name = node_name.split('.')[0] if '.' in node_name else node_name

        related_tables.add(table_name)

        for source, target, attr in self.graph.edges(data=True):
            if attr.get('type') == 'foreign_key':
                source_table = source.split('.')[0]
                target_table = target.split('.')[0]

                if table_name in (source_table, target_table):
                    related_tables.add(source_table)
                    related_tables.add(target_table)

        return related_tables

    def _get_table_ddl_from_schema(self, table_name: str) -> Optional[str]:
        try:
            statements = []
            current_stmt = []

            for line in self.ddl.split('\n'):
                current_stmt.append(line)
                if line.strip().endswith(';'):
                    statements.append('\n'.join(current_stmt))
                    current_stmt = []

            table_patterns = [
                f'create table `{table_name}`',
                f'create table {table_name}',
            ]

            for stmt in statements:
                stmt_lower = stmt.lower()
                if any(pattern in stmt_lower for pattern in table_patterns):
                    return stmt.strip()

        except Exception as e:
            self.logger.error(f"Error reading schema.sql: {str(e)}")
        return None

    def _reconstruct_ddl_for_node(self, node_name: str) -> Optional[str]:
        if node_name not in self.graph:
            return None

        table_name = node_name.split('.')[0] if '.' in node_name else node_name

        ddl = self._get_table_ddl_from_schema(table_name)
        if ddl:
            return ddl

        if self.graph.nodes[node_name].get('type') == 'table':
            columns = [n for n in self.graph.neighbors(node_name)]
            ddl = f"CREATE TABLE {node_name} (\n"
            column_defs = []
            for col in columns:
                definition = self.graph.nodes[col].get('definition', col)
                column_defs.append(f"    {definition}")
            ddl += ",\n".join(column_defs)
            ddl += "\n);"
            return ddl

        return None

    async def query_graph(self, query: str):
        self.logger.info(f"Starting graph query: {query}")
        matches = await self._query_graph_async(query)
        query_result_str = ""

        for match in matches:
            node_name = match[0]
            related_tables = self._get_related_tables(node_name)
            self.logger.debug(f"Related tables found for {node_name}: {related_tables}")

            for table in related_tables:
                table_ddl = self._reconstruct_ddl_for_node(table)
                if table_ddl:
                    query_result_str += f"\n{table_ddl}"

        self.logger.info("Graph query completed successfully")
        return query_result_str
