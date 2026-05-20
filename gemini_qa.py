"""
GROQ Question Answering Module
Generates answers using GROQ API with retrieved context.
"""

from groq import Groq
from typing import List, Tuple, Dict, Optional
import os


class GeminiQA:
    """Generates answers using GROQ API with retrieved context."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize GROQ QA system.
        
        Args:
            api_key: GROQ API key (defaults to GROQ_API_KEY environment variable)
            model: GROQ model to use (default: llama-3.3-70b-versatile)
        """
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set. "
                "Please set it or pass api_key parameter."
            )
        
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def generate_answer(
        self,
        question: str,
        retrieved_context: List[Tuple[str, str, float]],
        max_tokens: int = 1000
    ) -> Dict[str, str]:
        """
        Generate an answer using GROQ with retrieved context.
        
        Args:
            question: The question to answer
            retrieved_context: List of tuples (source_path, content, distance)
            max_tokens: Maximum tokens for the response
            
        Returns:
            Dictionary with answer, sources, and reasoning
        """
        # Build prompt with context
        context_text = self._format_context(retrieved_context)
        
        prompt = f"""You are a helpful assistant answering questions about Butterbot AI Research company.
        
Use the following context to answer the question. If the answer is not in the context, say "Information not found in provided documents."

CONTEXT:
{context_text}

QUESTION: {question}

Please provide:
1. A clear, concise answer
2. Your reasoning (especially for complex questions requiring synthesis)

Format your response as:
ANSWER: [Your answer here]
REASONING: [Your reasoning here]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering questions about Butterbot AI Research company."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            answer_text = response.choices[0].message.content
            
            # Parse response
            return self._parse_response(answer_text, retrieved_context)
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "reasoning": "Failed to generate answer due to API error"
            }
    
    def _format_context(self, retrieved_context: List[Tuple[str, str, float]]) -> str:
        """Format retrieved context for the prompt."""
        formatted = ""
        for idx, (source, content, distance) in enumerate(retrieved_context, 1):
            # Truncate content if too long
            truncated = content[:500] if len(content) > 500 else content
            formatted += f"\n[Source {idx}: {source}]\n{truncated}\n"
        
        return formatted
    
    def _parse_response(
        self,
        response_text: str,
        retrieved_context: List[Tuple[str, str, float]]
    ) -> Dict[str, str]:
        """
        Parse GROQ response into structured format.
        
        Args:
            response_text: Raw response from GROQ
            retrieved_context: Retrieved context for source attribution
            
        Returns:
            Dictionary with answer, sources, and reasoning
        """
        # Extract answer and reasoning
        answer = response_text
        reasoning = ""
        
        # Try to extract sections
        if "ANSWER:" in response_text:
            parts = response_text.split("ANSWER:")
            answer = parts[1].split("REASONING:")[0].strip() if len(parts) > 1 else response_text
        
        if "REASONING:" in response_text:
            parts = response_text.split("REASONING:")
            reasoning = parts[1].strip() if len(parts) > 1 else ""
        
        # Extract sources from retrieved context (first element of 3-element tuple)
        sources = [source for source, _, _ in retrieved_context]
        
        return {
            "answer": answer,
            "sources": sources,
            "reasoning": reasoning
        }
    
    def validate_answer(self, answer: Dict[str, str]) -> bool:
        """
        Validate that answer has required fields.
        
        Args:
            answer: Answer dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["answer", "sources"]
        return all(field in answer for field in required_fields)


if __name__ == "__main__":
    # Test Gemini QA
    qa = GeminiQA()
    
    test_context = [
        ("corporate/about-butterbot.html", "Butterbot is an AI research company founded in 2021."),
        ("corporate/company-values.md", "Our tagline is 'In Pursuit of Purpose'")
    ]
    
    result = qa.generate_answer(
        "What is Butterbot's tagline?",
        test_context
    )
    
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("Reasoning:", result["reasoning"])
