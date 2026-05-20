"""
Main QA System Orchestrator
Coordinates document ingestion, vector search, and answer generation.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from document_ingestion import DocumentIngestor
from vector_db import VectorDatabaseManager
from gemini_qa import GeminiQA


class ButterBotQASystem:
    """Main question-answering system for Butterbot AI Research."""
    
    def __init__(
        self,
        dataset_dir: str = "dataset",
        persist_dir: str = "./chroma_data",
        candidate_name: str = "AI Assistant",
        email: str = "assistant@butterbot.ai"
    ):
        """
        Initialize the QA system.
        
        Args:
            dataset_dir: Path to dataset folder
            persist_dir: Path for Chroma persistence
            candidate_name: Submission candidate name
            email: Submission email
        """
        self.dataset_dir = dataset_dir
        self.persist_dir = persist_dir
        self.candidate_name = candidate_name
        self.email = email
        
        self.ingestor = None
        self.db_manager = None
        self.qa_system = None
        self.documents = None
    
    def setup(self) -> None:
        """Set up the QA system."""
        print("=" * 60)
        print("Setting up Butterbot QA System")
        print("=" * 60)
        
        # Step 1: Ingest documents
        print("\n[1/3] Ingesting documents...")
        self.ingestor = DocumentIngestor(self.dataset_dir)
        self.documents = self.ingestor.ingest_all_documents()
        print(f"✓ Ingested {len(self.documents)} documents")
        
        # Step 2: Setup vector database
        print("\n[2/3] Setting up vector database...")
        self.db_manager = VectorDatabaseManager(self.persist_dir)
        self.db_manager.create_collection()
        self.db_manager.add_documents(self.documents)
        self.db_manager.persist()
        
        # Step 3: Initialize Gemini QA
        print("\n[3/3] Initializing Gemini QA system...")
        self.qa_system = GeminiQA()
        print("✓ Gemini QA system ready")
        
        print("\n" + "=" * 60)
        print("Setup complete! System ready to answer questions.")
        print("=" * 60)
    
    def answer_question(
        self,
        question_id: str,
        question: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a single question.
        
        Args:
            question_id: Question ID (e.g., "Q001")
            question: Question text
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Answer dictionary with answer, sources, and reasoning
        """
        # Retrieve relevant context
        retrieved = self.db_manager.search(question, top_k=top_k)
        
        # Generate answer
        result = self.qa_system.generate_answer(question, retrieved)
        
        # Format as required output
        return {
            "question_id": question_id,
            "answer": result["answer"],
            "sources": result["sources"],
            "reasoning": result.get("reasoning", "")
        }
    
    def answer_all_questions(self, questions_file: str) -> List[Dict[str, Any]]:
        """
        Answer all questions from the questions file.
        
        Args:
            questions_file: Path to questions.json
            
        Returns:
            List of answer dictionaries
        """
        # Load questions
        with open(questions_file, "r") as f:
            data = json.load(f)
        
        questions_list = data.get("questions", [])
        
        print("\n" + "=" * 60)
        print(f"Answering {len(questions_list)} questions...")
        print("=" * 60)
        
        answers = []
        for idx, q in enumerate(questions_list, 1):
            question_id = q["id"]
            question_text = q["question"]
            
            print(f"\n[{idx}/{len(questions_list)}] {question_id}: {question_text[:60]}...")
            
            try:
                answer = self.answer_question(question_id, question_text)
                answers.append(answer)
                print(f"✓ Answered {question_id}")
            except Exception as e:
                print(f"✗ Error answering {question_id}: {str(e)}")
                answers.append({
                    "question_id": question_id,
                    "answer": f"Error: {str(e)}",
                    "sources": [],
                    "reasoning": ""
                })
        
        return answers
    
    def save_answers(
        self,
        answers: List[Dict[str, Any]],
        output_file: str = "answers.json"
    ) -> None:
        """
        Save answers to JSON file in required format.
        
        Args:
            answers: List of answer dictionaries
            output_file: Output file path
        """
        submission = {
            "candidate_name": self.candidate_name,
            "submission_date": datetime.now().strftime("%Y-%m-%d"),
            "email": self.email,
            "answers": answers
        }
        
        with open(output_file, "w") as f:
            json.dump(submission, f, indent=2)
        
        print(f"\n✓ Answers saved to {output_file}")


def main():
    """Main entry point."""
    # Get the parent directory (TR_New)
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / "dataset"
    questions_file = script_dir.parent / "questions.json"
    
    # Initialize system
    system = ButterBotQASystem(
        dataset_dir=str(dataset_dir),
        candidate_name="Butterbot QA System",
        email="qa-system@butterbot.ai"
    )
    
    # Setup
    system.setup()
    
    # Answer all questions
    answers = system.answer_all_questions(str(questions_file))
    
    # Save results to parent directory
    output_file = script_dir.parent / "answers.json"
    system.save_answers(answers, str(output_file))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Submission Summary")
    print("=" * 60)
    print(f"Total questions answered: {len(answers)}")
    print(f"Output file: answers.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
