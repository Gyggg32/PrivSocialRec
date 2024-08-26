import praw
import configparser
import json
import os
import random
import csv
from datetime import datetime

# 加载配置
config = configparser.ConfigParser()
config.read('secrets.ini')

# 设置Reddit API凭证
reddit = praw.Reddit(
    user_agent=config.get('reddit', 'user_agent'),
    client_id=config.get('reddit', 'client_id'),
    client_secret=config.get('reddit', 'client_api_key')
)

# 保存用户信息（性别、年龄）的字典
user_info_cache = {}


# 随机生成用户性别和年龄，并缓存结果
def generate_user_info(username):
    if username not in user_info_cache:
        genders = ['Male', 'Female', 'Non-binary', 'Other']
        age_groups = ['1', '18', '25', '35', '45', '55']
        user_info_cache[username] = {
            'gender': random.choice(genders),
            'age_group': random.choice(age_groups)
        }
    return user_info_cache[username]



# 函数：抓取数据

def scrape_data(num_users=100, num_posts_per_user=5, output_file='data/dataset.json'):
    data = []
    processed_post_ids = set()
    posts = []
    submission_limit = num_users * num_posts_per_user * 2
    attempts = 0
    max_attempts = 5

    # 尝试从不同来源获取帖子
    sources = ['top', 'hot', 'new']
    while len(posts) == 0 and attempts < max_attempts:
        try:
            for source in sources:
                submissions = getattr(reddit.subreddit('all'), source)(limit=submission_limit)
                for submission in submissions:
                    if len(posts) >= num_users * num_posts_per_user:
                        break

                    post_id = submission.id
                    subreddit_name = submission.subreddit.display_name

                    if post_id not in processed_post_ids:
                        posts.append({
                            'post_id': post_id,
                            'subreddit': subreddit_name,
                            'title': submission.title,
                            'url': submission.url,
                            'created_utc': submission.created_utc,
                            'num_comments': submission.num_comments
                        })
                        processed_post_ids.add(post_id)

                if len(posts) >= num_users * num_posts_per_user:
                    break

            if len(posts) == 0:
                print(f"第 {attempts + 1} 次尝试未获取到帖子，重试中...")
                attempts += 1
        except Exception as e:
            print(f"获取帖子时发生错误: {e}")
            attempts += 1

    if len(posts) == 0:
        print("警告: 未能获取任何帖子。")
        return

    if len(posts) < num_users * num_posts_per_user:
        print(f"警告: 可用帖子数量不足 {num_users * num_posts_per_user} 个，实际数量为 {len(posts)} 个。")

    # 为每个用户生成数据
    usernames = [comment.author.name for comment in
                 reddit.subreddit('all').comments(limit=num_users * num_posts_per_user) if comment.author]

    for username in random.sample(usernames, min(num_users, len(usernames))):
        user_info = generate_user_info(username)
        # 确保样本大小不超过帖子列表长度
        random_posts = random.sample(posts, min(num_posts_per_user, len(posts)))

        for post_data in random_posts:
            post_id = post_data['post_id']
            submission = reddit.submission(id=post_id)
            user_comments = [
                comment for comment in submission.comments.list()
                if hasattr(comment, 'author') and comment.author and comment.author.name == username
            ]
            score = 0

            user_data = {
                'username': username,
                'gender': user_info['gender'],
                'age_group': user_info['age_group'],
                'post_id': post_id,
                'subreddit': post_data['subreddit'],
                'recent_comment': random.choice([True, False]),  # 随机生成最近评论
                'created_utc': post_data['created_utc'],
                'num_comments': post_data['num_comments'],
                'user_rating': score,
                'title': post_data['title'],
                'url': post_data['url']
            }

            data.append(user_data)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"数据已保存到 {output_file}")


def json_to_csv(json_file, user_csv_file, post_csv_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(user_csv_file), exist_ok=True)
    os.makedirs(os.path.dirname(post_csv_file), exist_ok=True)

    with open(user_csv_file, 'w', newline='', encoding='utf-8') as user_csv:
        user_writer = csv.writer(user_csv)
        user_writer.writerow(['Index', 'Username', 'Gender', 'Age Group'])

        with open(post_csv_file, 'w', newline='', encoding='utf-8') as post_csv:
            post_writer = csv.writer(post_csv)
            post_writer.writerow(
                ['Index', 'Post ID', 'User Rating', 'Subreddit', 'Title', 'URL', 'Created UTC', 'Num Comments'])

            for index, entry in enumerate(data):
                if 'username' in entry:
                    user_writer.writerow([index, entry['username'], entry['gender'], entry['age_group']])
                post_writer.writerow([
                    index, entry['post_id'], entry['user_rating'], entry['subreddit'], entry['title'], entry['url'],
                    entry['created_utc'], entry['num_comments']
                ])

    print(f"数据已写入 {user_csv_file} 和 {post_csv_file}")

if __name__ == "__main__":

    scrape_data(num_users=200, num_posts_per_user=5)
    json_to_csv('data/dataset.json', 'data/users.csv', 'data/posts.csv')
    print("Completed!")
