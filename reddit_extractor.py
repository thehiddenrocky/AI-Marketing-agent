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

    try:
        for post in subreddit.new(limit=100):
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

            post.comments.replace_more(limit=None)
            print(f"Fetching comments for post ID {post.id}...")
            for comment in post.comments.list():
                comment_info = {
                    'id': comment.id,
                    'author': str(comment.author),
                    'body': comment.body,
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
            time.sleep(0.5)

    except Exception as e:
        print(f"Error fetching posts from r/{subreddit_name}: {e}")

    return posts_data

def get_relevant_topics(query):
    """
    Use OpenAI to generate relevant search phrases for a query
    """
    prompt = f"""
        You are a Reddit search expert who understands how Redditors discuss and search for topics.
        Given the query "{query}", generate 5 search phrases that will find the most relevant subreddits.

        Consider:
        - How Redditors naturally phrase their questions/discussions
        - Common abbreviations and terminology used on Reddit
        - Related tools, technologies, or concepts frequently discussed
        - Industry-specific subreddit naming patterns
        - Problem-focused search terms (as many discussions are about solving problems)

        For example, if query is "project management software":
        - projectmanagement (direct community)
        - asana vs trello (tool comparison commonly discussed)
        - agile tools (methodology + tools)
        - jira alternatives (tool alternative discussions)
        - remote team management (broader problem space)
        Return exactly 5 search phrases, one per line.
        Focus on phrases that would lead to active, relevant subreddit communities.
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
        
        print(f"Generated search phrases: {search_phrases}")
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
        if post["selftext"]:
            chunks = text_splitter.split_text(post["selftext"]) if len(post["selftext"]) > 500 else [post["selftext"]]

            chunk_embeddings = embeddings_client.embed_documents(chunks)

            for chunk, embedding in zip(chunks, chunk_embeddings):
                documents.append(chunk)
                metadatas.append({
                    "id": post["id"],
                    "type": "post",
                    "title": post["title"],
                    "author": post["author"],
                    "url": post["url"],
                    "created_utc": str(post["created_utc"])
                })
                ids.append(f"doc_{doc_id}")
                doc_id += 1

        for comment in post.get("comments", []):
            if comment["body"]:
                chunks = text_splitter.split_text(comment["body"]) if len(comment["body"]) > 500 else [comment["body"]]

                chunk_embeddings = embeddings_client.embed_documents(chunks)

                for chunk, embedding in zip(chunks, chunk_embeddings):
                    documents.append(chunk)
                    metadatas.append({
                        "id": comment["id"],
                        "type": "comment",
                        "post_id": post["id"],
                        "post_title": post["title"],
                        "author": comment["author"],
                        "parent_id": comment["parent_id"],
                        "created_utc": str(comment["created_utc"])
                    })
                    ids.append(f"doc_{doc_id}")
                    doc_id += 1

        if len(documents) >= 500:
            embeddings = embeddings_client.embed_documents(documents)
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added batch of {len(documents)} documents to Chroma")
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
        print(f"Added final batch of {len(documents)} documents to Chroma")

    return collection.count()


def main():
    query = input("Enter the search query for subreddits: ")
    subreddit_limit = int(input("Enter the number of subreddits to find: "))

    search_terms = get_relevant_topics(query)
    found_subreddits = search_subreddits(search_terms, limit=subreddit_limit)

    if not found_subreddits:
        print("No subreddits with enough subscribers found with the given query.")
        return

    print("\nRetrieving subreddit information...")
    subreddit_infos = []
    for subreddit_name in found_subreddits:
        info = get_subreddit_info(subreddit_name)
        if info:
            subreddit_infos.append(info)
        time.sleep(1)

    if not subreddit_infos:
        print("No subreddit information retrieved.")
        return

    sorted_subreddits = sorted(subreddit_infos, key=lambda x: x['subscribers'], reverse=True)

    print("\nRanked Subreddits:")
    for idx, info in enumerate(sorted_subreddits, start=1):
        print(f"{idx}. r/{info['name']} - {info['subscribers']} subscribers")

    for info in sorted_subreddits:
        if info['subscribers'] > 50000:
            subreddit_name = info['name']
            print(f"\nProcessing r/{subreddit_name}...")

            posts = get_subreddit_posts(subreddit_name)

            data_to_save = {
                'subreddit_info': info,
                'posts': posts
            }

            filename = f"{subreddit_name}_data.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=4)
                print(f"Data from r/{subreddit_name} saved to {filename}")
            except Exception as e:
                print(f"Error saving data for r/{subreddit_name}: {e}")

            try:
                collection_name = f"reddit_{subreddit_name.lower()}"
                doc_count = process_and_store_in_chroma(posts, collection_name)
                print(f"Processed and stored {doc_count} documents in Chroma collection '{collection_name}'")
            except Exception as e:
                print(f"Error processing and storing data in Chroma: {e}")

            time.sleep(2)

    print("Data collection, processing, and embedding completed.")


if __name__ == '__main__':
    main()
