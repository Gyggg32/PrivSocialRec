import praw
import configparser
import random
import json
import string
import os

# Import NLTK and download necessary resources (run this once)
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys

# Import configuration parameters, user agent for PRAW Reddit object
config = configparser.ConfigParser()
config.read('secrets.ini')

# Load user agent string
reddit_user_agent = config.get('reddit', 'user_agent')
client_id = config.get('reddit', 'client_id')
client_secret = config.get('reddit', 'client_api_key')


# Main data scrapping script
def scrape_data(n_scrape_loops=10, dataset='train'):
    """This is the main function that runs the scrapping functionality through praw."""


    # Initialize PRAW Reddit object
    r = praw.Reddit(user_agent=reddit_user_agent, client_id=client_id, client_secret=client_secret)

    # Prepare text processing tools
    translate_table = dict((ord(char), None) for char in string.punctuation)
    stop = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Define file paths
    data_file_path = 'data/' + dataset + '_reddit_data.json'
    users_file_path = 'data/scrapped_users.json'

    # Ensure data directory exists
    os.makedirs(os.path.dirname(data_file_path), exist_ok=True)

    # Initialize data structures
    if os.path.exists(data_file_path):
        with open(data_file_path, 'r') as data_file:
            reddit_data = json.load(data_file)
    else:
        reddit_data = []

    if os.path.exists(users_file_path):
        with open(users_file_path, 'r') as data_file:
            scrapped_users = json.load(data_file)
    else:
        scrapped_users = []

    # Data scraping
    for scrape_loop in range(n_scrape_loops):
        try:
            all_comments = r.subreddit('all').comments(limit=50)
            print("Scrape Loop " + str(scrape_loop))
            for cmt in all_comments:
                user = cmt.author
                if user:
                    print("Collecting Data for User " + user.name)
                    if user.name in scrapped_users:
                        print('User ' + user.name + ' already scraped')
                    else:
                        scrapped_users.append(user.name)
                        for user_comment in user.comments.new(limit=20):
                                reddit_data.append([
                                    user.name,
                                    user_comment.subreddit.display_name,
                                    user_comment.created_utc
                                ])
        except Exception as e:
            print(e)

    # Save the data
    with open(data_file_path, 'w') as data_file:
        json.dump(reddit_data, data_file)

    with open(users_file_path, 'w') as data_file:
        json.dump(scrapped_users, data_file)


if __name__ == "__main__":
    scrape_data(10, 'train')
