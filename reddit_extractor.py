import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
import chromadb
import praw
import json
import time

load_dotenv()


CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
USER_AGENT = os.getenv('REDDIT_USER_AGENT')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)


def search_subreddits(search_terms, limit=50):
    """
    Search for subreddits matching multiple search terms.
    
    Args:
        search_terms (list): List of search queries to use
        limit (int): Maximum number of subreddits to return per search term
    
    Returns:
        list: List of unique subreddit names found across all search terms
    """
    print(f"Searching for subreddits matching {len(search_terms)} search terms...")
    subreddits = set()  # Using a set to avoid duplicates
    
    try:
        for term in search_terms:
            print(f"Searching with term: '{term}'...")
            term_subreddits = []  
            for subreddit in reddit.subreddits.search(term, limit=limit):
                subreddits.add(subreddit.display_name)
                term_subreddits.append(subreddit.display_name)
            print(f"Found with '{term}': {term_subreddits}")
    except Exception as e:
        print(f"Error during subreddit search: {e}")
    
    return list(subreddits)  # Convert set back to list for consistency


def get_subreddit_info(subreddit_name):
    """
    Retrieve information about a subreddit.
    """
    try:
        subreddit = reddit.subreddit(subreddit_name)
        return {
            'name': subreddit.display_name,
            'title': subreddit.title,
            'subscribers': subreddit.subscribers,
            'public_description': subreddit.public_description
        }
    except Exception as e:
        print(f"Error retrieving info for r/{subreddit_name}: {e}")
        return None

def get_subreddit_posts(subreddit_name):
    """
    Fetch all posts and comments from a subreddit, and extract problem-related sentences.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    print(f"Fetching posts from r/{subreddit_name}...")
    
    # Search terms for telecom-related posts
    search_terms = [
        "Elisa", "DNA", "Telia", "Lounea",
        "internet provider", "mobile provider",
        "broadband", "internet speed",
        "fiber optic", "5G", "4G"
    ]
    
    try:
        # Search for each term
        for term in search_terms:
            print(f"Searching for posts containing '{term}'...")
            
            # Use Reddit's search function
            for post in subreddit.search(term, limit=100, sort='relevance'):
                if post.id not in [p['id'] for p in posts_data]:  # Avoid duplicates
                    post_info = {
                        'id': post.id,
                        'title': post.title,
                        'author': str(post.author),
                        'score': post.score,
                        'upvotes': post.ups,
                        'downvotes': post.downs,
                        'num_comments': post.num_comments,
                        'created_utc': post.created_utc,
                        'url': post.url,
                        'permalink': post.permalink,
                        'selftext': post.selftext,
                        'comments': []
                    }
                    
                    # Fetch comments
                    post.comments.replace_more(limit=None)
                    print(f"Fetching comments for post ID {post.id}...")
                    for comment in post.comments.list():
                        if hasattr(comment, 'body') and comment.body:
                            comment_info = {
                                'id': comment.id,
                                'body': comment.body,
                                'author': str(comment.author),
                                'score': comment.score,
                                'upvotes': comment.ups,
                                'downvotes': comment.downs,
                                'created_utc': comment.created_utc,
                                'parent_id': comment.parent_id,
                                'link_id': comment.link_id,
                                'permalink': comment.permalink
                            }
                            post_info['comments'].append(comment_info)
                    
                    posts_data.append(post_info)
                    time.sleep(1)  # Be nice to Reddit's API
        
        print(f"Total posts fetched: {len(posts_data)}")
        return posts_data
        
    except Exception as e:
        print(f"Error fetching posts: {e}")
        return posts_data

def get_relevant_topics(query):
    """
    Use OpenAI to generate relevant search phrases for marketing research
    """
    prompt = f"""
        You are an expert marketing research analyst specializing in competitive intelligence and consumer insights.
        Given the query "{query}", generate 5 search phrases that will help uncover valuable market insights.
        
        Consider these key marketing research aspects:
        1. Consumer Sentiment & Voice of Customer:
           - Customer complaints and praise
           - Service quality perceptions
           - Price sensitivity discussions
           
        2. Competitive Intelligence:
           - Competitor comparisons
           - Market positioning
           - Service differentiators
           
        3. Market Trends:
           - Emerging technologies
           - Industry innovations
           - Consumer behavior shifts
           
        4. Product/Service Analysis:
           - Feature comparisons
           - Quality assessments
           - Value propositions
           
        5. Brand Perception:
           - Brand reputation
           - Customer loyalty factors
           - Service reliability
        
        For example, if query is "DNA broadband":
        - DNA vs Elisa broadband speed comparison
        - DNA internet customer service experience
        - DNA fiber optic coverage areas review
        - DNA broadband pricing complaints
        - DNA internet reliability issues
        
        Return exactly 5 search phrases, one per line.
        Focus on phrases that real customers would use in discussions.
        Ensure phrases will find relevant discussions about market positioning, competitive advantages, and customer pain points.
        Do not include any bullets, numbers, or prefixes.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=256
        )
        
        # Extract and clean phrases
        search_phrases = [
            phrase.strip()
            for phrase in response.choices[0].message.content.split('\n')
            if phrase.strip() and not phrase.startswith(('-', '*', '•', '1', '2', '3', '4', '5'))
        ]
        
        print(f"Generated marketing research phrases:\n" + "\n".join(f"- {phrase}" for phrase in search_phrases))
        return search_phrases
    except Exception as e:
        print(f"Error generating topics: {e}")
        return [query]  # Fallback to original query

def process_and_store_in_chroma(posts_data: List[Dict], collection_name: str):
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    embeddings_client = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    collection = chroma_client.get_or_create_collection(name=collection_name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    documents = []
    metadatas = []
    ids = []
    doc_id = 0

    print(f"Processing posts and comments for {collection_name}...")

    for post in posts_data:
        # Always include the title as a document
        post_content = f"Title: {post['title']}\n\n"
        if post["selftext"]:
            post_content += post["selftext"]
        
        # Split the combined content
        chunks = text_splitter.split_text(post_content) if len(post_content) > 500 else [post_content]
        chunk_embeddings = embeddings_client.embed_documents(chunks)

        for chunk, embedding in zip(chunks, chunk_embeddings):
            documents.append(chunk)
            metadatas.append({
                "id": post["id"],
                "type": "post",
                "title": post["title"],
                "author": post["author"],
                "url": post["url"],
                "created_utc": str(post["created_utc"]),
                "score": post["score"],
                "num_comments": post["num_comments"]
            })
            ids.append(f"doc_{doc_id}")
            doc_id += 1

        # Process comments
        for comment in post.get("comments", []):
            if comment["body"]:
                comment_chunks = text_splitter.split_text(comment["body"]) if len(comment["body"]) > 500 else [comment["body"]]
                comment_embeddings = embeddings_client.embed_documents(comment_chunks)

                for chunk, embedding in zip(comment_chunks, comment_embeddings):
                    documents.append(chunk)
                    metadatas.append({
                        "id": comment["id"],
                        "type": "comment",
                        "post_id": post["id"],
                        "post_title": post["title"],
                        "author": comment["author"],
                        "created_utc": str(comment["created_utc"]),
                        "score": comment["score"]
                    })
                    ids.append(f"doc_{doc_id}")
                    doc_id += 1

        # Add documents in batches to avoid memory issues
        if len(documents) >= 500:
            embeddings = embeddings_client.embed_documents(documents)
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            documents = []
            metadatas = []
            ids = []

    # Add any remaining documents
    if documents:
        embeddings = embeddings_client.embed_documents(documents)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    print(f"Added {doc_id} documents to collection {collection_name}")
    return collection.count()


def process_subreddit(subreddit_name: str):
    """
    Process a single subreddit: fetch its data and store in both JSON and ChromaDB
    """
    print(f"\nProcessing r/{subreddit_name}...")
    
    # Get subreddit info
    subreddit_info = get_subreddit_info(subreddit_name)
    if not subreddit_info:
        print(f"Could not retrieve information for r/{subreddit_name}")
        return
    
    # Get posts and comments
    posts = get_subreddit_posts(subreddit_name)
    if not posts:
        print(f"No posts found in r/{subreddit_name}")
        return
        
    # Save to JSON
    data_to_save = {
        'subreddit_info': subreddit_info,
        'posts': posts
    }
    
    filename = f"{subreddit_name}_data.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving JSON data: {e}")
    
    # Store in ChromaDB
    try:
        collection_name = f"reddit_{subreddit_name.lower()}"
        doc_count = process_and_store_in_chroma(posts, collection_name)
        print(f"Stored {doc_count} documents in ChromaDB collection '{collection_name}'")
    except Exception as e:
        print(f"Error storing in ChromaDB: {e}")


def main():
    subreddit_name = input("Enter the subreddit name (without r/): ").strip()
    process_subreddit(subreddit_name)


if __name__ == '__main__':
    main()
