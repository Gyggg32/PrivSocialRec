import praw
import configparser

# 加载配置
config = configparser.ConfigParser()
config.read('secrets.ini')

# 设置Reddit API凭证
reddit = praw.Reddit(
    user_agent=config.get('reddit', 'user_agent'),
    client_id=config.get('reddit', 'client_id'),
    client_secret=config.get('reddit', 'client_api_key')
)


def get_top_subreddits(limit=10):
    """获取热门的subreddit供用户选择"""
    top_subreddits = []
    for subreddit in reddit.subreddits.popular(limit=limit):
        top_subreddits.append(subreddit.display_name)
    return top_subreddits


def get_subreddit_posts(subreddit_name, sort_by="hot", limit=5):
    """获取指定subreddit中的帖子，可以选择获取热门或最新帖子"""
    subreddit = reddit.subreddit(subreddit_name)
    if sort_by == "hot":
        posts = subreddit.hot(limit=limit)
    elif sort_by == "new":
        posts = subreddit.new(limit=limit)

    post_list = []
    for post in posts:
        post_info = {
            "title": post.title,
            "url": post.url,
            "score": post.score,
            "comments": post.num_comments,
            "created": post.created_utc
        }
        post_list.append(post_info)
    return post_list


def user_cold_start():
    """用户冷启动过程"""
    print("欢迎！请选择你感兴趣的话题：")
    subreddits = get_top_subreddits()

    for idx, subreddit in enumerate(subreddits, 1):
        print(f"{idx}. {subreddit}")

    # 用户选择感兴趣的subreddit
    selected_indices = input("请输入你感兴趣的话题编号（用逗号分隔多个选择）：")
    selected_indices = [int(i.strip()) for i in selected_indices.split(',')]

    selected_subreddits = [subreddits[i - 1] for i in selected_indices]

    print(f"你选择了以下话题：{', '.join(selected_subreddits)}")
    return selected_subreddits


def recommend_posts_to_user(subreddits, sort_by="hot", limit=5):
    """根据用户选择的话题为其推荐帖子"""
    print("\n根据你选择的子话题，我们为你推荐以下帖子：")

    for subreddit in subreddits:
        print(f"\n推荐来自 /r/{subreddit} 的帖子：")
        posts = get_subreddit_posts(subreddit, sort_by=sort_by, limit=limit)
        for idx, post in enumerate(posts, 1):
            print(f"{idx}. {post['title']} (得分: {post['score']}, 评论数: {post['comments']})")
            print(f"链接: {post['url']}\n")


# 启动用户冷启动和推荐过程
if __name__ == "__main__":
    user_topics = user_cold_start()

    # 用户选择推荐方式：热门或最新帖子
    sort_preference = input("你希望我们推荐最新的帖子还是最热门的帖子？（输入 'new' 或 'hot'）：").strip().lower()
    if sort_preference not in ['new', 'hot']:
        sort_preference = 'hot'  # 默认推荐热门帖子

    # 根据用户选择的subreddit推荐帖子
    recommend_posts_to_user(user_topics, sort_by=sort_preference)
