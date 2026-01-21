import uuid
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Core Dependencies
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Professional Formatting
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
import logging

# --- CONFIGURATION ---
COLLECTION_NAME = "aegis_production_memory"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Setup structured logging
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
log = logging.getLogger("rich")
console = Console()

class AegisEngine:
    """
    Core engine for the Aegis Multimodal Disaster Response System.
    Handles Vector Storage (Qdrant), Neural Encoding, and Semantic Retrieval.
    """

    def __init__(self):
        log.info("[bold cyan]Initializing Aegis System Core...[/bold cyan]")
        
        # 1. Initialize Vector Database
        # using :memory: for demonstration; strictly equivalent to Qdrant Cloud in architecture
        self.client = QdrantClient(":memory:")
        
        # 2. Initialize Neural Encoder
        # Using a lightweight transformer for efficient CPU inference
        log.info(f"Loading embedding model: [green]{EMBEDDING_MODEL}[/green]")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        
        # 3. Setup Schema
        self._initialize_collection()

    def _initialize_collection(self):
        """Ensures the Qdrant collection exists with correct vector configuration."""
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384,  # Matching MiniLM dimensions
                    distance=models.Distance.COSINE
                )
            )
            log.info("Vector collection created successfully.")

    def ingest_data(self, data_stream: List[Dict[str, Any]]):
        """
        Ingests multimodal metadata into the vector store.
        
        Args:
            data_stream: List of dictionaries containing 'content', 'type', 'severity', etc.
        """
        points = []
        log.info(f"Ingesting batch of {len(data_stream)} records...")

        for item in data_stream:
            # Generate Embedding
            vector = self.encoder.encode(item["content"]).tolist()
            
            # Create Payload (Metadata)
            # Simulating timestamps to demonstrate temporal filtering capabilities
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "type": item["type"],
                    "location": item["location"],
                    "severity": item["severity"],
                    "content": item["content"],
                    "timestamp": datetime.now().isoformat(),
                    "source": item.get("source", "system_ingest")
                }
            ))

        # Bulk Upsert for performance
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        log.info("[bold green]Ingestion Complete.[/bold green] Knowledge base updated.")

    def search_and_reason(self, query: str, top_k: int = 5):
        """
        Executes a semantic search query with severity-based re-ranking logic.
        
        Args:
            query: Natural language query string.
            top_k: Number of results to retrieve.
        """
        log.info(f"Processing Query: [bold]'{query}'[/bold]")
        
        # 1. Encode Query
        query_vector = self.encoder.encode(query).tolist()
        
        # 2. Retrieve from Qdrant
        hits = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k * 2, # Fetch more to allow for re-ranking
            with_payload=True
        )

        # 3. Apply Re-ranking Logic (Business Logic Layer)
        # Prioritize 'critical' severity items if semantic scores are close
        reranked_results = self._apply_business_logic(hits)
        
        # 4. Display Results
        self._render_results(reranked_results[:top_k])

    def _apply_business_logic(self, hits):
        """Applies domain-specific weighting to search results."""
        processed_hits = []
        for hit in hits:
            score = hit.score
            severity = hit.payload.get("severity", "low")
            
            # Boost score for critical items (Algorithm adjustments)
            if severity == "critical":
                score += 0.15
            
            processed_hits.append((score, hit))
        
        # Sort by new adjusted score
        return sorted(processed_hits, key=lambda x: x[0], reverse=True)

    def _render_results(self, scored_hits):
        """Renders search results in a structured table."""
        table = Table(title="Retrieval Results", border_style="grey50")
        table.add_column("Score", justify="right", style="cyan", no_wrap=True)
        table.add_column("Severity", justify="center")
        table.add_column("Type", style="magenta")
        table.add_column("Location", style="green")
        table.add_column("Content", style="white")

        for score, hit in scored_hits:
            severity_style = "bold red" if hit.payload['severity'] == "critical" else "yellow"
            
            table.add_row(
                f"{score:.4f}",
                f"[{severity_style}]{hit.payload['severity'].upper()}[/{severity_style}]",
                hit.payload['type'],
                hit.payload['location'],
                hit.payload['content']
            )
        console.print(table)

# --- RUNTIME EXECUTION ---

if __name__ == "__main__":
    # Initialize System
    engine = AegisEngine()

    # Mock Data Injection (Simulating real-time feed)
    mock_data = [
        {"type": "social_media", "location": "Sector 4", "severity": "critical", "content": "Urgent: First floor flooded at Apollo Hospital, patients stuck."},
        {"type": "drone_log", "location": "Sector 4", "severity": "critical", "content": "Image analysis: Road blocked by debris near Hospital entry."},
        {"type": "iot_sensor", "location": "Dam Gate", "severity": "high", "content": "Water level critical. Overflow warning triggered."},
        {"type": "call_log", "location": "Sector 9", "severity": "low", "content": "Tree down on main street, traffic diverted."},
        {"type": "logistics", "location": "Sector 2", "severity": "medium", "content": "Food supplies deployed to community center."},
    ]
    
    engine.ingest_data(mock_data)

    # Interactive CLI Loop
    console.print(Panel("[bold]Aegis System Ready.[/bold] Enter query (or 'exit'):", style="green"))
    
    while True:
        try:
            user_input = console.input("\n[bold]Query > [/bold]")
            if user_input.lower() in ["exit", "quit"]:
                log.info("Terminating session.")
                sys.exit(0)
            
            if user_input.strip():
                engine.search_and_reason(user_input)
                
        except KeyboardInterrupt:
            sys.exit(0)