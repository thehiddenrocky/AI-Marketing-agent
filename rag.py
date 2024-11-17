import json
import warnings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
import string

warnings.filterwarnings("ignore")

class AnalysisTemplate:
    """Handles loading and accessing analysis templates."""
    
    def __init__(self, template_path: str):
        """Initialize with path to template JSON file."""
        self.template = self._load_template(template_path)
        
    def _load_template(self, path: str) -> Dict:
        """Load template from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_enabled_analysis_types(self) -> List[str]:
        """Get list of enabled analysis types."""
        return [
            analysis_type
            for analysis_type, config in self.template["analysis_types"].items()
            if config.get("enabled", True)
        ]
    
    def get_analyses_for_type(self, analysis_type: str) -> Dict:
        """Get all analyses configured for a specific type."""
        return self.template["analysis_types"][analysis_type]["analyses"]
    
    def get_search_query(self, analysis_type: str, analysis_name: str, **kwargs) -> str:
        """Get search query with replacements."""
        query_template = self.template["analysis_types"][analysis_type]["analyses"][analysis_name]["search_query"]
        return string.Template(query_template).safe_substitute(**kwargs)
    
    def generate_prompt(self, analysis_type: str, analysis_name: str, context: str, **kwargs) -> str:
        """Generate analysis prompt using template."""
        analysis_config = self.template["analysis_types"][analysis_type]["analyses"][analysis_name]
        base_prompt = analysis_config["base_prompt"]
        guide = analysis_config["extraction_guide"]
        
        prompt = f"""{base_prompt}

Context:
{context}

Please focus on identifying:
{json.dumps(guide.get('look_for', []), indent=2)}

"""
        
        # Add specific guide questions if they exist
        for key, question in guide.items():
            if key not in ['look_for', 'output_format'] and isinstance(question, str):
                prompt += f"{key}: {question}\n"
        
        prompt += f"""
Return results in this format:
{json.dumps(guide['output_format'], indent=2)}
"""
        
        return string.Template(prompt).safe_substitute(**kwargs)

class SubredditContent:
    """Handles loading and processing subreddit data."""
    
    def __init__(self, data_path: str):
        """Initialize with path to subreddit JSON file."""
        self.data = self._load_data(data_path)
        
    def _load_data(self, path: str) -> Dict:
        """Load subreddit data from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_processed_documents(self) -> List[str]:
        """Process posts into documents."""
        documents = []
        for post in self.data['posts']:
            text = (
                f"Title: {post['title']}\n"
                f"Content: {post.get('selftext', '')}\n"
                f"Score: {post['score']}\n"
                f"Comments: {post['num_comments']}\n"
                f"URL: {post['url']}\n"
            )
            documents.append(text)
        return documents

class VectorDB:
    """Handles vector store operations."""
    
    def __init__(self, embeddings_model: str = "text-embedding-3-large"):
        """Initialize vector store components."""
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
    
    def create_and_store(self, documents: List[str]):
        """Create chunks and build vector store."""
        chunks = self.text_splitter.create_documents(documents)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """Perform similarity search."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def save(self, path: str):
        """Save vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(path)
    
    def load(self, path: str):
        """Load vector store from disk."""
        self.vector_store = FAISS.load_local(path, self.embeddings)

class Analyzer:
    """Handles analysis using OpenAI."""
    
    def __init__(self, openai_api_key: str):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=openai_api_key)
    
    def analyze_context(self, prompt: str) -> Dict:
        """Analyze context using OpenAI."""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.choices[0].message.content)

class SubredditRAGSystem:
    """Main RAG system combining all components."""
    
    def __init__(self, template_path: str, data_path: str, openai_api_key: str):
        """Initialize RAG system components."""
        self.template = AnalysisTemplate(template_path)
        self.content = SubredditContent(data_path)
        self.vector_db = VectorDB()
        self.analyzer = Analyzer(openai_api_key)
        
        # Process and store content
        documents = self.content.get_processed_documents()
        self.vector_db.create_and_store(documents)
    
    def run_analysis(self, analysis_type: str, analysis_name: str, **kwargs) -> Dict:
        """Run a specific type of analysis."""
        # Get search query and run search
        search_query = self.template.get_search_query(analysis_type, analysis_name, **kwargs)
        context = "\n\n".join(self.vector_db.search(search_query))
        
        # Generate prompt and run analysis
        prompt = self.template.generate_prompt(analysis_type, analysis_name, context, **kwargs)
        return self.analyzer.analyze_context(prompt)
    
    def run_all_enabled_analyses(self) -> Dict[str, Dict]:
        """Run all enabled analyses defined in the template."""
        results = {}
        
        for analysis_type in self.template.get_enabled_analysis_types():
            results[analysis_type] = {}
            analyses = self.template.get_analyses_for_type(analysis_type)
            
            for analysis_name in analyses.keys():
                # Handle special cases where we need results from previous analyses
                if analysis_name.startswith("analyze_"):
                    base_name = f"find_{analysis_name.split('_')[1]}"
                    if base_name in analyses:
                        items = results[analysis_type][base_name]
                        detailed_results = []
                        
                        for item in items.get("tools", []) + items.get("trends", []):
                            kwargs = {
                                f"{analysis_type.split('_')[0]}_name": item["name"]
                            }
                            result = self.run_analysis(analysis_type, analysis_name, **kwargs)
                            detailed_results.append(result)
                            
                        results[analysis_type][analysis_name] = detailed_results
                else:
                    results[analysis_type][analysis_name] = self.run_analysis(
                        analysis_type, analysis_name
                    )
        
        return results
    
    def save_vector_store(self, path: str):
        """Save vector store for later use."""
        self.vector_db.save(path)
    
    def load_vector_store(self, path: str):
        """Load existing vector store."""
        self.vector_db.load(path)

def print_analysis_results(results: Dict[str, Dict]):
    """Pretty print analysis results."""
    for analysis_type, type_results in results.items():
        print(f"\n{'='*50}")
        print(f"Analysis Type: {analysis_type}")
        print(f"{'='*50}")
        
        for analysis_name, analysis_results in type_results.items():
            print(f"\n{'-'*40}")
            print(f"Analysis: {analysis_name}")
            print(f"{'-'*40}")
            
            if isinstance(analysis_results, list):
                for idx, result in enumerate(analysis_results, 1):
                    print(f"\nItem {idx}:")
                    print(json.dumps(result, indent=2))
            else:
                print(json.dumps(analysis_results, indent=2))

def main():
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Initialize RAG system
    rag = SubredditRAGSystem(
        template_path="simple-augment-templates.json",
        data_path="data/LeadGeneration_data.json",
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Run all enabled analyses
    results = rag.run_all_enabled_analyses()
    
    # Print results
    print_analysis_results(results)
    
    # Save vector store
    rag.save_vector_store("assets/vector_store")

if __name__ == "__main__":
    main()