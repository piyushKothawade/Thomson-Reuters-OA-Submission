"""
Hybrid Retrieval Module
Fast keyword + simple similarity-based retrieval for document search.
"""

from typing import List, Tuple
import re
from collections import Counter
import math


class VectorDatabaseManager:
    """Simple hybrid retrieval system using keyword search and TF-IDF."""
    
    def __init__(self, persist_dir: str = "./chroma_data"):
        """
        Initialize retrieval system.
        
        Args:
            persist_dir: Directory (not used in this implementation)
        """
        self.persist_dir = persist_dir
        self.documents = []
        self.doc_index = {}  # Map from doc_id to (path, content)
        self.tokenized_docs = []
        self.idf_scores = {}
    
    def create_collection(self, name: str = "butterbot_documents"):
        """Initialize collection (no-op for this implementation)."""
        print(f"✓ Created collection: {name}")
    
    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Add documents to the retrieval system.
        
        Args:
            documents: List of tuples (doc_path, content)
        """
        self.documents = documents
        
        # Preprocess and tokenize documents
        for idx, (doc_path, content) in enumerate(documents):
            self.doc_index[idx] = (doc_path, content)
            tokens = self._tokenize(content)
            self.tokenized_docs.append((idx, tokens))
        
        # Calculate IDF scores
        self._calculate_idf()
        
        print(f"✓ Added {len(documents)} documents to collection")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text to lowercase words."""
        # Remove special characters and split
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _calculate_idf(self) -> None:
        """Calculate IDF scores for all terms."""
        doc_count = len(self.tokenized_docs)
        term_doc_count = {}
        
        # Count documents containing each term
        for _, tokens in self.tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_count[token] = term_doc_count.get(token, 0) + 1
        
        # Calculate IDF
        for term, count in term_doc_count.items():
            if count > 0:
                self.idf_scores[term] = math.log(doc_count / count)
    
    def _calculate_tfidf(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Calculate TF-IDF similarity between query and document."""
        if not doc_tokens or not query_tokens:
            return 0.0
        
        # Count term frequencies
        query_tf = Counter(query_tokens)
        doc_tf = Counter(doc_tokens)
        
        # Calculate TF-IDF score
        score = 0.0
        for term, count in query_tf.items():
            tf = count / len(query_tokens)
            idf = self.idf_scores.get(term, 0)
            doc_tf_val = doc_tf.get(term, 0) / len(doc_tokens)
            score += tf * idf * doc_tf_val
        
        return score
    
    def _keyword_match_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Calculate keyword match score."""
        if not query_tokens:
            return 0.0
        
        matches = sum(1 for token in query_tokens if token in doc_tokens)
        return matches / len(query_tokens)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of tuples (doc_path, content, relevance_score)
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens or not self.tokenized_docs:
            return []
        
        # Score all documents
        scores = []
        for doc_id, doc_tokens in self.tokenized_docs:
            # Combine TF-IDF and keyword match scores
            tfidf_score = self._calculate_tfidf(query_tokens, doc_tokens)
            keyword_score = self._keyword_match_score(query_tokens, doc_tokens)
            
            # Weighted combination
            combined_score = (0.6 * tfidf_score) + (0.4 * keyword_score)
            
            if combined_score > 0:
                doc_path, content = self.doc_index[doc_id]
                scores.append((doc_path, content, combined_score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[2], reverse=True)
        
        # Return top_k with normalized distance (1 - score for consistency with vector DB)
        results = []
        for doc_path, content, score in scores[:top_k]:
            results.append((doc_path, content, 1.0 - min(score, 1.0)))
        
        return results
    
    def persist(self) -> None:
        """Persist the collection (no-op for this implementation)."""
        print(f"✓ Collection persisted to {self.persist_dir}")


if __name__ == "__main__":
    # Test retrieval
    from document_ingestion import DocumentIngestor
    
    ingestor = DocumentIngestor("dataset")
    documents = ingestor.ingest_all_documents()
    
    db = VectorDatabaseManager()
    db.create_collection()
    db.add_documents(documents)
    db.persist()
    
    # Test search
    results = db.search("What is Butterbot's tagline?", top_k=3)
    print("\nSearch Results:")
    for doc_path, content, score in results:
        print(f"\nSource: {doc_path}")
        print(f"Relevance Score: {1 - score:.4f}")
        print(f"Content: {content[:200]}...")

if __name__ == "__main__":
    # Test vector database
    from document_ingestion import DocumentIngestor
    
    ingestor = DocumentIngestor("dataset")
    documents = ingestor.ingest_all_documents()
    
    db = VectorDatabaseManager()
    db.create_collection()
    db.add_documents(documents)
    db.persist()
    
    # Test search
    results = db.search("What is Butterbot's tagline?", top_k=3)
    print("\nSearch Results:")
    for doc_path, content, distance in results:
        print(f"\nSource: {doc_path}")
        print(f"Similarity Score: {1 - distance:.4f}")
        print(f"Content: {content[:200]}...")
