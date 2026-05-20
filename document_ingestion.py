"""
Document Ingestion Module
Handles loading and parsing documents in various formats from the dataset folder.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict
import html2text
from bs4 import BeautifulSoup


class DocumentIngestor:
    """Ingests and parses documents from the dataset folder."""
    
    def __init__(self, dataset_dir: str):
        """Initialize with dataset directory path."""
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    def ingest_all_documents(self) -> List[Tuple[str, str]]:
        """
        Ingest all documents from the dataset folder.
        
        Returns:
            List of tuples (relative_path, content)
        """
        documents = []
        
        # Walk through all files in dataset directory
        for file_path in self.dataset_dir.rglob("*"):
            if file_path.is_file():
                try:
                    relative_path = file_path.relative_to(self.dataset_dir)
                    content = self._parse_file(file_path)
                    if content.strip():  # Only include non-empty documents
                        documents.append((str(relative_path), content))
                        print(f"✓ Ingested: {relative_path}")
                except Exception as e:
                    print(f"✗ Failed to ingest {file_path}: {str(e)}")
        
        return documents
    
    def _parse_file(self, file_path: Path) -> str:
        """
        Parse a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parsed text content
        """
        suffix = file_path.suffix.lower()
        
        if suffix == ".md":
            return self._read_markdown(file_path)
        elif suffix == ".html":
            return self._read_html(file_path)
        elif suffix == ".csv":
            return self._read_csv(file_path)
        elif suffix in [".txt", ".text"]:
            return self._read_text(file_path)
        elif suffix == ".json":
            return self._read_json(file_path)
        else:
            # Try to read as text
            return self._read_text(file_path)
    
    def _read_markdown(self, file_path: Path) -> str:
        """Read markdown file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    
    def _read_html(self, file_path: Path) -> str:
        """Read and convert HTML to text."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        
        # Use html2text for conversion
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        text = h.handle(html_content)
        
        # Also try BeautifulSoup for fallback
        if not text.strip():
            soup = BeautifulSoup(html_content, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
        
        return text
    
    def _read_csv(self, file_path: Path) -> str:
        """Read CSV file and convert to readable text."""
        rows = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(" | ".join(row))
        
        return "\n".join(rows)
    
    def _read_text(self, file_path: Path) -> str:
        """Read plain text file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    
    def _read_json(self, file_path: Path) -> str:
        """Read JSON file and convert to readable text."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        
        # Convert JSON to formatted string
        return json.dumps(data, indent=2)


if __name__ == "__main__":
    # Test document ingestion
    ingestor = DocumentIngestor("dataset")
    docs = ingestor.ingest_all_documents()
    print(f"\nTotal documents ingested: {len(docs)}")
