"""Main RAG pipeline: orchestrates all components."""
from PIL import Image

from src.data_loader import load_openi_from_huggingface, save_reports_to_json, load_reports_from_json
from src.chunking import chunk_reports
from src.vector_store import index_chunks
from src.retriever import retrieve, RetrievalResult
from src.generator import generate_answer, GenerationResult
import config


class MedicalImagingRAG:
    """
    Medical Imaging Multimodal RAG Pipeline.

    Usage:
        rag = MedicalImagingRAG()
        rag.ingest(max_samples=200)     # One-time: load data + build index
        result = rag.query("What are common findings in chest X-rays?")
        print(result.answer)
    """

    def __init__(self):
        self._is_ingested = False
        self._check_index()

    def _check_index(self):
        """Check if vector store already has data."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            collection = client.get_or_create_collection(name=config.COLLECTION_NAME)
            count = collection.count()
            if count > 0:
                self._is_ingested = True
                print(f"Found existing index with {count} documents.")
        except Exception:
            pass

    def ingest(self, max_samples: int = 300):
        """
        Load data and build the vector index.

        Args:
            max_samples: Number of reports to load from the dataset
        """
        reports_path = config.DATASET_DIR / "reports.json"
        if not reports_path.exists():
            # Fallback for benchmark datasets that use passages.json
            passages_path = config.DATASET_DIR / "passages.json"
            if passages_path.exists():
                reports_path = passages_path

        if reports_path.exists():
            print(f"Loading reports from {reports_path} ...")
            reports = load_reports_from_json(reports_path)
        else:
            reports = load_openi_from_huggingface(max_samples=max_samples)
            reports_path.parent.mkdir(parents=True, exist_ok=True)
            save_reports_to_json(reports, reports_path)

        # Chunk reports
        chunks = chunk_reports(reports)

        # Index into vector store
        index_chunks(chunks)

        self._is_ingested = True
        print("\nIngestion complete! Ready to answer queries.")

    def query(
        self,
        question: str,
        image: Image.Image | None = None,
        top_k: int = None,
    ) -> GenerationResult:
        """
        Query the RAG system.

        Args:
            question: User's text question
            image: Optional medical image (e.g., X-ray)
            top_k: Number of documents to retrieve

        Returns:
            GenerationResult with answer and sources
        """
        if not self._is_ingested:
            raise RuntimeError(
                "No data indexed yet. Run rag.ingest() first."
            )

        # Retrieve relevant documents
        retrieval = retrieve(question, image=image, top_k=top_k)

        # Generate answer
        result = generate_answer(retrieval, image=image)

        return result
