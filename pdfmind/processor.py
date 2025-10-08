import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

class PDFProcessor:
    """
    A utility class for processing PDF files, splitting into text chunks,
    and generating embeddings for offline vector retrieval.
    """

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.reader = PdfReader(pdf_path)
        self.text = ""
        self.chunks = []
        self.embeddings = None

    def extract_text(self):
        """Extract all text content from the PDF file."""
        self.text = ""
        for page in self.reader.pages:
            self.text += page.extract_text() + "\n"
        return self.text

    @staticmethod
    def split_text_smart(text, chunk_size=500, overlap=50):
        """Split text intelligently into overlapping chunks."""
        if not text:
            return []

        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        initial_chunks = []
        current_chunk = ""

        for sentence in sentences:
            estimated_length = len(current_chunk) + len(sentence) + 1
            if estimated_length <= chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    initial_chunks.append(current_chunk.strip())
                if len(sentence) > chunk_size:
                    initial_chunks.extend(
                        [sentence[i:i + chunk_size].strip() for i in range(0, len(sentence), chunk_size)]
                    )
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            initial_chunks.append(current_chunk.strip())

        # Add overlaps
        final_chunks = []
        for i, chunk in enumerate(initial_chunks):
            if i > 0:
                overlap_text = initial_chunks[i - 1][-overlap:].strip()
                if not chunk.startswith(overlap_text):
                    chunk = overlap_text + " " + chunk
            final_chunks.append(chunk.strip())

        return final_chunks

    def count_pages(self):
        """Return the total number of pages in the PDF."""
        return len(self.reader.pages)

    def generate_embeddings(self, model_name="all-MiniLM-L6-v2", chunk_size=500, overlap=50):
        """Generate embeddings for PDF chunks."""
        print("🔍 Extracting text...")
        text = self.extract_text()

        print("✂️ Splitting text into chunks...")
        self.chunks = self.split_text_smart(text, chunk_size, overlap)

        print(f"🧠 Generating embeddings using model '{model_name}'...")
        model = SentenceTransformer(model_name)
        self.embeddings = model.encode(self.chunks, show_progress_bar=True)
        return self.embeddings


def process_pdf_and_store_in_chroma(pdf_path, db_path="./chroma_db", collection_name="pdf_collection"):
    """
    Full offline RAG pipeline:
    1. Extract & chunk PDF text
    2. Generate embeddings
    3. Store in a persistent local ChromaDB
    """
    try:
        processor = PDFProcessor(pdf_path)
        embeddings = processor.generate_embeddings()

        print("💾 Initializing ChromaDB...")
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name=collection_name)

        print("📥 Adding embeddings to ChromaDB...")
        ids = [f"chunk_{i}" for i in range(len(processor.chunks))]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=processor.chunks
        )

        print(f"✅ Stored {len(processor.chunks)} chunks in local DB at: {db_path}")
        return processor, collection

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None, None
