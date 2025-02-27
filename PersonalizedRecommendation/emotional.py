import pandas as pd
import numpy as np
import os
import random
import tkinter as tk
from tkinter import simpledialog, messagebox
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# 本地BERT模型路径
model_name = "/home/quinn/scretflow/BERT"

# 加载 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertForSequenceClassification.from_pretrained(model_name, from_pt=True)

# 加载用户搜索记录
def load_search_history(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

# 将文本数据转换为 BERT 输入格式
def encode(texts, tokenizer, max_length=128):
    inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True, max_length=max_length)
    return inputs['input_ids'], inputs['attention_mask']

# 获取情感分数
def get_sentiment_scores(model, tokenizer, texts):
    input_ids, attention_mask = encode(texts, tokenizer)
    logits = model({'input_ids': input_ids, 'attention_mask': attention_mask})[0]
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    return probs

# 转换情感分数到等级
def score_to_label(score):
    if score < 0.2:
        return -1
    elif score < 0.4:
        return -0.75
    elif score < 0.6:
        return -0.25
    elif score < 0.8:
        return 0
    elif score < 1.0:
        return 0.25
    elif score < 1.2:
        return 0.75
    else:
        return 1

# 对用户的搜索历史进行情感分析，并计算综合评分
def analyze_sentiment(search_history_path, user_input=None):
    search_history = load_search_history(search_history_path)
    texts = [query.strip() for query in search_history if query.strip()]
    
    # 将用户输入添加到分析中
    if user_input:
        texts.append(user_input.strip())

    sentiment_scores = get_sentiment_scores(bert_model, tokenizer, texts)
    labels = [score_to_label(np.max(score)) for score in sentiment_scores]
    
    # 计算综合评分（可以选择平均值或其他加权方法）
    final_score = np.mean(labels)
    
    return labels, final_score

def ask_user_for_counseling():
    root = tk.Tk()
    root.geometry("800x400")  
    root.withdraw()  # 隐藏主窗口


    response = messagebox.askyesno("Counseling", "Do you need professional psychological counseling from our platform?")
    
    if response:
        messagebox.showinfo("Counseling", 
                            "We are always here to help you.\nPlatform Counseling Hotline: 12320, \nFree Counseling Website: https://www.psychologicalcounselingcenter.com/")
    else:
        messagebox.showinfo("Counseling", 
                            "We are always here to help you. \nThis is not your fault, and you are not alone. Many people are experiencing the same pain as you.")
    
    root.destroy()  # 关闭窗口

# 推荐帖子集合
def recommend_posts():
    recommend_files = [
        "/home/quinn/Documents/recommend1.txt",
        "/home/quinn/Documents/recommend2.txt",
        "/home/quinn/Documents/recommend3.txt"
    ]
    # 随机选择一个推荐文件
    selected_file = random.choice(recommend_files)

    # 从选定的推荐文件中读取内容
    with open(selected_file, 'r') as file:
        posts = file.readlines()

    # 返回文件中的所有帖子
    return [post.strip() for post in posts if post.strip()]

# 获取用户输入
def get_user_input():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    user_input = simpledialog.askstring("User Input", "What would you like to search for?")
    root.destroy()  # 关闭窗口
    return user_input

# 根据情感分析结果进行推荐
def recommend_based_on_sentiment():
    recommended_posts = recommend_posts()
    return recommended_posts
    
# 示例使用
search_history_path = "/home/quinn/Documents/user_search_history.txt"
user_input = get_user_input()  # 获取用户输入
labels, final_score = analyze_sentiment(search_history_path, user_input)

# 打印分析结果
print(f"Final Sentiment Score: {final_score:.2f}")

# 如果检测到负面情绪，则推荐帖子
if final_score < 0:
    print("Recommended Posts:")
    posts = recommend_based_on_sentiment()
    for post in posts:
        print(post)
    ask_user_for_counseling()
else:
    print("No negative sentiment detected.")
