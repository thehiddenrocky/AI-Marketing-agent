import requests
from typing import Dict, List, Optional
import time
from config import REDDIT_CONFIG

class RedditAPI:
    def __init__(self):
        self.client_id = REDDIT_CONFIG['client_id']
        self.client_secret = REDDIT_CONFIG['client_secret']
        self.user_agent = REDDIT_CONFIG['user_agent']
        self.auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        self.token = None
        self.token_expiry = 0

    def _get_token(self) -> None:
        """Get OAuth token from Reddit"""
        try:
            data = {
                'grant_type': 'client_credentials'
            }
            headers = {'User-Agent': self.user_agent}
            
            print("Attempting to authenticate...")
            response = requests.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=self.auth,
                data=data,
                headers=headers
            )
            
            if response.status_code == 200:
                self.token = response.json()['access_token']
                self.token_expiry = time.time() + 3600  # Token expires in 1 hour
                print("Authentication successful!")
            else:
                print(f"Authentication failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()
                
        except Exception as e:
            print(f"Error during authentication: {str(e)}")
            raise

    def _ensure_valid_token(self) -> None:
        if not self.token or time.time() > self.token_expiry:
            self._get_token()

    def get_subreddit_posts(self, subreddit: str, limit: int = 25) -> List[Dict]:
        """Get posts from a subreddit"""
        self._ensure_valid_token()
        
        headers = {
            'User-Agent': self.user_agent,
            'Authorization': f'Bearer {self.token}'
        }
        
        params = {
            'limit': min(limit, 100)
        }
        
        response = requests.get(
            f'https://oauth.reddit.com/r/{subreddit}/hot',
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            return [post['data'] for post in response.json()['data']['children']]
        else:
            print(f"Error fetching posts: {response.status_code}")
            print(f"Response: {response.text}")
            response.raise_for_status()

def main():
    reddit = RedditAPI()
    
    try:
        posts = reddit.get_subreddit_posts('python', limit=5)
        
        print("\nSuccessfully retrieved posts from r/python:")
        for i, post in enumerate(posts, 1):
            print(f"\n{i}. {post['title']}")
            print(f"   Score: {post['score']}")
            print(f"   Comments: {post['num_comments']}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()