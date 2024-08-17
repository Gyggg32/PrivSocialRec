import praw
import configparser
import json
import os

# Load configuration from secrets.ini
config = configparser.ConfigParser()
config.read('secrets.ini')

# Set up Reddit API credentials
reddit = praw.Reddit(
    user_agent=config.get('reddit', 'user_agent'),
    client_id=config.get('reddit', 'client_id'),
    client_secret=config.get('reddit', 'client_api_key')
)

# Function to scrape data from subreddits
def scrape_subreddit_data(subreddits, num_posts=10, output_file='post1.json'):
    data = []
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Scraping subreddit: {subreddit_name}")

        for post in subreddit.hot(limit=num_posts):
            post_data = {
                'subreddit_name': subreddit_name,  # Add subreddit name here
                'title': post.title,
                'id': post.id,
                'score': post.score,
                'created_at': post.created_utc,  # Keep the original UNIX timestamp
                'url': post.url,
                'num_comments': post.num_comments
            }
            data.append(post_data)

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save data to JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data saved to {output_file}")

# Function to get a list of popular subreddits
def get_popular_subreddits(limit=10):
    popular_subreddits = []
    for subreddit in reddit.subreddits.popular(limit=limit):
        popular_subreddits.append(subreddit.display_name)
    return popular_subreddits

if __name__ == "__main__":
    # Get a list of popular subreddits
    subreddits = get_popular_subreddits(limit=300)  # Adjust the limit as needed
    scrape_subreddit_data(subreddits, num_posts=3, output_file='data/post1.json')
