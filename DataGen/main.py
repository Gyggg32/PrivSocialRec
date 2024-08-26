import json
from datetime import datetime

# 加载数据
with open('dataset.json', 'r') as f:
    data = json.load(f)

# 建立user_id到subreddit的计数映射
user_subreddit_count = {}

# 遍历所有数据记录，统计每个user_id在每个subreddit中的帖子数量
for entry in data:
    username = entry['username']
    subreddit = entry['subreddit']

    if username not in user_subreddit_count:
        user_subreddit_count[username] = {}

    if subreddit not in user_subreddit_count[username]:
        user_subreddit_count[username][subreddit] = 0

    user_subreddit_count[username][subreddit] += 1

# 函数：计算新的user_rating
def calculate_user_rating(entry, user_subreddit_count):
    user_id = entry['username']
    subreddit = entry['subreddit']
    num_comments = entry['num_comments']
    created_utc = entry['created_utc']

    # 计算subreddit活跃度分数
    subreddit_activity_score = user_subreddit_count[user_id][subreddit]

    # 计算时间衰减分数（时间越久远，得分越低）
    post_age_days = (datetime.utcnow() - datetime.utcfromtimestamp(created_utc)).days
    time_decay_score = 1 / (1 + 0.01 * post_age_days)  # 修改时间衰减公式，使其逐渐减小但不至于为 0

    # 计算评论数得分
    comment_count_score = num_comments / 1000.0

    # 综合评分
    score = int((subreddit_activity_score * 6) + (time_decay_score * 1) + (comment_count_score * 3))

    # 打印详细信息
    print(f"user_rating = {subreddit_activity_score} * 6 + {time_decay_score*10} * 2 + {comment_count_score} * 2 = {score}")

    return round(score, 2)

# 遍历所有数据记录，计算并更新user_rating
for entry in data:
    entry['user_rating'] = calculate_user_rating(entry, user_subreddit_count)

# 将新的数据写回到json文件
with open('updated_dataset.json', 'w') as f:
    json.dump(data, f, indent=4)

print("user_rating 已更新并保存到 updated_dataset.json 文件中")

