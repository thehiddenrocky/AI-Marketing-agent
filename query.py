import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
import chromadb
import time
import json
from openai import OpenAI

load_dotenv()


CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
USER_AGENT = os.getenv('REDDIT_USER_AGENT')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def query_chroma(query_text: str, collection_name: str, n_results: int = 5):
    """
    Query the Chroma database for similar content.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embeddings_client = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    collection = chroma_client.get_collection(name=collection_name)

    # Remove the augment_query call here since queries are already augmented
    query_embedding = embeddings_client.embed_query(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results

def query_and_format_results(query_text: str, collection_name: str, n_results: int = 5):
    """
    Query Chroma and return formatted results with all related information.

    Args:
        query_text (str): The search query
        collection_name (str): Name of the Chroma collection to search
        n_results (int): Number of results to return

    Returns:
        dict: Formatted results with query info and matched documents
    """
    try:
        # Get raw results from Chroma
        results = query_chroma(query_text, collection_name, n_results)

        # Format the results
        formatted_results = {
            "query_info": {
                "search_query": query_text,
                "collection": collection_name,
                "total_results": len(results['ids'][0])
            },
            "matches": []
        }

        # Process each result
        for i in range(len(results['ids'][0])):
            match = {
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            }
            formatted_results["matches"].append(match)

        return formatted_results

    except Exception as e:
        print(f"Error querying database: {e}")
        return None


def print_search_results(formatted_results):
    """
    Print search results in a readable format.
    """
    if not formatted_results:
        print("No results found.")
        return

    print("\n=== Search Results ===")
    print(f"Query: '{formatted_results['query_info']['search_query']}'")
    print(f"Collection: {formatted_results['query_info']['collection']}")
    print(f"Total matches: {formatted_results['query_info']['total_results']}")
    print("\n=== Matches ===")

    for idx, match in enumerate(formatted_results['matches'], 1):
        print(f"\n--- Match {idx} ---")

        # Print metadata
        metadata = match['metadata']
        if metadata['type'] == 'post':
            print(f"Type: Post")
            print(f"Title: {metadata['title']}")
        else:
            print(f"Type: Comment")
            print(f"On Post: {metadata['post_title']}")

        print(f"Author: u/{metadata['author']}")
        print(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(metadata['created_utc'])))}")

        if 'url' in metadata:
            print(f"URL: {metadata['url']}")

        # Print content
        print("\nContent:")
        print("-" * 80)
        print(match['content'])
        print("-" * 80)

        if match['distance'] is not None:
            print(f"Relevance Score: {1 - match['distance']:.2%}")

        print("\n")

def augment_query(query_text: str):
    """Augment query using augment.json"""
    try:
        with open('augment.json', 'r') as f:
            rules = json.load(f)
            return [f"{rule.get('prefix', '')} {query_text} {rule.get('suffix', '')}".strip() 
                    for rule in rules.get('searches', [])]
    except:
        return [query_text]


def generate_answer(results, query: str):
    """
    Generate an answer based on search results using OpenAI.
    
    Args:
        results (dict): Formatted search results from query_and_format_results
        query (str): Original user query
    
    Returns:
        str: Generated answer
    """
    try:
        # Load prompt template from JSON
        with open('prompt_config.json', 'r') as f:
            config = json.load(f)
        
        # Format context from results
        context_parts = []
        for match in results['matches']:
            context_parts.append(f"Content: {match['content']}\n")
            
        context = "\n".join(context_parts)
        print ("\n\n Before formatting prompt")
        print("context: ", context)
        print("query: ", query)
        print("instructions: ", config['instructions'])
        # Format the complete prompt
        prompt = config['prompt_template'].format(
            context=context,
            query=query,
            instructions=config['instructions']
        )
        
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Generate response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=256
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None

# Example usage:
if __name__ == '__main__':
    # Example 1: Query existing data
    print("=== Example Query Usage ===")

    while True:
        # Get available collections
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        collections = chroma_client.list_collections()

        if not collections:
            print("No collections found in the database.")
            break

        print("\nAvailable collections:")
        for idx, collection in enumerate(collections, 1):
            print(f"{idx}. {collection.name}")

        # Get user input
        collection_choice = input("\nEnter collection number (or 'q' to quit): ")
        if collection_choice.lower() == 'q':
            break

        try:
            collection_name = collections[int(collection_choice) - 1].name
        except (ValueError, IndexError):
            print("Invalid collection number.")
            continue

        # Get search query
        query = input("\nEnter your search query: ")
        n_results = input("Number of results to return (default 5): ")
        n_results = int(n_results) if n_results.isdigit() else 5

        # Perform search
        print("\nSearching...")
        augmented_queries = augment_query(query)

        print("augmented queries: ", augmented_queries)
        for augmented_query in augmented_queries:
            print(f"\n=== Search augmented query: {augmented_query} ===")
            results = query_and_format_results(augmented_query, collection_name, n_results)
            answer = generate_answer(results, augmented_query)
            print_search_results(results)
            if answer:
                print("\n=== Generated Answer ===")
                print(answer)

        another = input("\nWould you like to do another search? (y/n): ")
        if another.lower() != 'y':
            break