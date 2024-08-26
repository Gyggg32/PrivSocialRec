import json

# 加载 JSON 文件中的数据
with open('updated_dataset.json', 'r') as file:
    data = json.load(file)

# 创建存储每个 .dat 文件的唯一条目的集合或列表
ratings_data = []
users_data = set()
posts_data = set()

# 定义age_group和num_comments的范围处理
def process_age_group(age_group):
    age = int(age_group)
    if 0 <= age <= 1:
        return "1"
    elif 1 < age <= 18:
        return "18"
    elif 18 < age <= 25:
        return "25"
    elif 25 < age <= 35:
        return "35"
    elif 35 < age <= 45:
        return "45"
    elif 45 < age <= 50:
        return "50"
    else:
        return "56"

def process_num_comments(num_comments):
    comments = int(num_comments)
    if 0 <= comments <= 1:
        return "1"
    elif 1 < comments <= 1000:
        return "1000"
    elif 1000 < comments <= 4000:
        return "4000"
    elif 4000 < comments <= 8000:
        return "8000"
    elif 8000 < comments <= 10000:
        return "10000"
    else:
        return "12000"

# 处理数据集
for entry in data:
    username = entry['username']
    gender = entry['gender']
    age_group = process_age_group(entry['age_group'])
    post_id = entry['post_id']
    user_rating = entry['user_rating']
    created_utc = entry['created_utc']
    title = entry['title']
    num_comments = process_num_comments(entry['num_comments'])

    # 为 ratings.dat 准备数据
    ratings_data.append(f"{username}::{post_id}::{user_rating}::{created_utc}")

    # 为 users.dat 准备数据
    users_data.add(f"{username}::{gender}::{age_group}")

    # 为 posts.dat 准备数据
    posts_data.add(f"{post_id}::{title}::{num_comments}")

# 将数据写入 .dat 文件
# 将数据写入 .dat 文件，并使用 utf-8 编码
with open('ratings.dat', 'w', encoding='utf-8') as ratings_file:
    ratings_file.write("\n".join(ratings_data))

with open('users.dat', 'w', encoding='utf-8') as users_file:
    users_file.write("\n".join(users_data))

with open('posts.dat', 'w', encoding='utf-8') as posts_file:
    posts_file.write("\n".join(posts_data))


