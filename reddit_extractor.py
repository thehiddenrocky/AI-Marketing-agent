import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

load_dotenv()


CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
USER_AGENT = os.getenv('REDDIT_USER_AGENT')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

import praw
import json
import time

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
        # Fetch posts; adjust sorting and limit as needed
        for submission in subreddit.new(limit=100):
            post_info = {
                'id': submission.id,
                'title': submission.title,
                'author': str(submission.author),
                'score': submission.score,
                'upvotes': submission.ups,
                'downvotes': submission.downs,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'url': submission.url,
                'permalink': submission.permalink,
                'selftext': submission.selftext,
                'comments': []
            }

            # Fetch comments
            submission.comments.replace_more(limit=None)
            print(f"  Fetching comments for post ID {submission.id}...")
            for comment in submission.comments.list():
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

    '''for info in sorted_subreddits:
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

        time.sleep(2)

    print("\nData collection and extraction completed.")'''

if __name__ == '__main__':
    main()

