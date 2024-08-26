import secretflow as sf
print('The version of SecretFlow:{}'.format(sf.__version__))

sf.shutdown()
sf.init(['alice','bob'],address='local',log_to_driver=False)
alice,bob=sf.PYU('alice'),sf.PYU('bob')
#3
def load_data(filename,columns):
    data={}
    with open(filename,"r",encoding="unicode_escape") as f:
        for line in f:
            ls=line.strip("\n").split("::")
            data[ls[0]]=dict(zip(columns[1:],ls[1:]))
    return data
#4
fed_csv={alice:"alice_reddit.csv",bob:"bob_reddit.csv"}
csv_writer_container={alice:open(fed_csv[alice],"w"),bob:open(fed_csv[bob],"w")}
part_columns={
    alice:["UserName","Gender","Age"],
    bob:["PostID","UserRating","Title","CreatedUTC","NumComments"]
}
#5
for device,writer in csv_writer_container.items():
    writer.write("ID,"+",".join(part_columns[device])+"\n")
#6
f=open("/home/quinn/scretflow/DeepFM-final/ratings.dat","r",encoding="unicode_escape")

users_data=load_data(
    "/home/quinn/scretflow/DeepFM-final/users.dat",
    columns=["UserName","Gender","Age"],
)

posts_data=load_data("/home/quinn/scretflow/DeepFM-final/posts.dat",columns=["PostID","Title","NumComments"])
ratings_columns=["UserName","PostID","UserRating","CreatedUTC"]

rating_data=load_data("/home/quinn/scretflow/DeepFM-final/ratings.dat",columns=ratings_columns)

def _parse_example(feature,columns,index):
    if "Title" in feature.keys():
        feature["Title"]=feature["Title"].replace(",","_")
    values=[]
    values.append(str(index))
    for c in columns:
        values.append(feature[c])
    return ",".join(values)

index=0
num_sample=1000
for line in f:
    ls=line.strip().split("::")
    rating=dict(zip(ratings_columns,ls))
    rating.update(users_data.get(ls[0]))
    rating.update(posts_data.get(ls[1]))
    for device,columns in part_columns.items():
        parse_f=_parse_example(rating,columns,index)
        csv_writer_container[device].write(parse_f+"\n")
    index+=1
    if num_sample>0 and index>=num_sample:
        break
for w in csv_writer_container.values():
    w.close()
#7
! head alice_reddit.csv
#8
! head bob_reddit.csv
#9
def create_dataset_builder_alice(
        batch_size=128,
        repeat_count=5,
):
    def dataset_builder(x):
        import pandas as pd
        import tensorflow as tf

        x=[dict(t) if isinstance(t,pd.DataFrame) else t for t in x]
        x=x[0] if len(x)==1 else tuple(x)
        data_set=(
            tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(repeat_count)
        )

        return data_set

    return dataset_builder

def create_dataset_builder_bob(
        batch_size=128,
        repeat_count=5,
):
    def _parse_bob(row_sample,label):
        import tensorflow as tf

        y_t=label["UserRating"]
        y=tf.expand_dims(
            tf.where(
                y_t>3,
                tf.ones_like(y_t,dtype=tf.float32),
                tf.zeros_like(y_t,dtype=tf.float32),
            ),
            axis=1,
        )
        return row_sample,y

    def dataset_builder(x):
        import pandas as pd
        import tensorflow as tf

        x=[dict(t) if isinstance(t,pd.DataFrame) else t for t in x]
        x=x[0] if len(x)==1 else tuple(x)
        data_set=(
            tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(repeat_count)
        )

        data_set=data_set.map(_parse_bob)

        return data_set

    return dataset_builder

data_builder_dict={
    alice:create_dataset_builder_alice(
        batch_size=128,
        repeat_count=5,
    ),
    bob:create_dataset_builder_bob(
        batch_size=128,
        repeat_count=5,
    ),
}
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import pandas as pd
import tensorflow as tf

# 指定本地模型路径
model_name = '/home/quinn/scretflow/BERT'

# 加载帖子数据
posts_data = pd.read_csv("/home/quinn/scretflow/DeepFM-final/posts.dat", delimiter="::", names=["PostID", "Title", "NumComments"], engine='python')

# 加载BERT模型和分词器（本地路径），指定使用 PyTorch 权重
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertModel.from_pretrained(model_name, from_pt=True)

# BERT情感分析函数
def bert_encode(texts, tokenizer, max_len=128):
    encodings = tokenizer(
        texts.tolist(), 
        max_length=max_len, 
        truncation=True, 
        padding=True, 
        add_special_tokens=True,
        return_tensors='tf'
    )
    return encodings

# 4. 提取 BERT 特征并进行平均池化
def extract_bert_features(text_column):
    encodings = bert_encode(text_column, tokenizer)
    bert_outputs = bert_model(encodings).last_hidden_state
    # 使用平均池化将维度从 (batch_size, seq_len, hidden_size) 转换为 (batch_size, hidden_size)
    sentence_embeddings = tf.reduce_mean(bert_outputs, axis=1)
    # 如果需要一维标量，可以进一步减少维度
    sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)
    return sentence_embeddings

# 对帖子标题进行 BERT 编码并生成情感嵌入
bert_embeddings = extract_bert_features(posts_data['Title']).numpy()

# 将 BERT 嵌入向量存储为对象列
posts_data['bert_embeddings'] = list(bert_embeddings)
COMMENT_VOCAB=[1,1000,4000,8000,10000,15000]
#11
def create_base_model_alice():
    def create_model():
        import tensorflow as tf

        def preprocess():
            inputs = {
                "UserName": tf.keras.Input(shape=(1,), dtype=tf.string),
                "Gender": tf.keras.Input(shape=(1,), dtype=tf.string),
                "Age": tf.keras.Input(shape=(1,), dtype=tf.int64),
            }
            user_id_output = tf.keras.layers.Hashing(
                num_bins=NUM_USERS, output_mode="one_hot"
            )
            user_gender_output = tf.keras.layers.StringLookup(
                vocabulary=GENDER_VOCAB, output_mode="one_hot"
            )

            user_age_out = tf.keras.layers.IntegerLookup(
                vocabulary=AGE_VOCAB, output_mode="one_hot"
            )

            outputs = {
                "UserName": user_id_output(inputs["UserName"]),
                "Gender": user_gender_output(inputs["Gender"]),
                "Age": user_age_out(inputs["Age"]),
            }
            return tf.keras.Model(inputs=inputs, outputs=outputs)

        preprocess_layer = preprocess()
        model = DeepFMbase(
            dnn_units_size=[256, 32],
            preprocess_layer=preprocess_layer,
        )
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
        )
        return model

    return create_model
#12
def create_base_model_bob():
    def create_model():
        import tensorflow as tf

        def preprocess():
            inputs = {
                "PostID": tf.keras.Input(shape=(1,), dtype=tf.string),
                "NumComments": tf.keras.Input(shape=(1,), dtype=tf.int64),
                "BertEmbeddings": tf.keras.Input(shape=(768,), dtype=tf.float32), # 加入BERT情感嵌入
            }
            post_id_out = tf.keras.layers.Hashing(
                num_bins=NUM_POSTS, output_mode="one_hot"
            )

            post_comment_output = tf.keras.layers.IntegerLookup( 
                vocabulary=COMMENT_VOCAB, output_mode="one_hot"
            )

            # BERT嵌入层
            bert_output = tf.keras.layers.Dense(256, activation='relu')(inputs["BertEmbeddings"])

            outputs = {
                "PostID": post_id_out(inputs["PostID"]),
                "NumComments": post_comment_output(inputs["NumComments"]),
                "BertEmbeddings": bert_output,  # 加入BERT处理后的特征
            }
            return tf.keras.Model(inputs=inputs, outputs=outputs)

        preprocess_layer = preprocess()
        model = DeepFMbase(
            dnn_units_size=[256, 32],
            preprocess_layer=preprocess_layer,
        )
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
        )
        return model

    return create_model#13
def create_fuse_model():
    def create_model():
        import tensorflow as tf

        model=DeepFMfuse(dnn_units_size=[256,256,32])
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )
        return model

    return create_model
base_model_dict = {
    alice: create_base_model_alice(),
    bob: create_base_model_bob(),
}
model_fuse=create_fuse_model()
from secretflow.data.vertical import read_csv as v_read_csv
from secretflow.ml.nn import SLModel

builder_base_alice = create_base_model_alice()
builder_base_bob = create_base_model_bob()
builder_fuse = create_fuse_model()

sl_model = SLModel(
    base_model_dict={alice: builder_base_alice, bob: builder_base_bob},
    device_y=bob,
    model_fuse=builder_fuse,
)

vdf=v_read_csv(
    {alice:"alice_reddit.csv",bob:"bob_reddit.csv"},keys="ID",drop_keys="ID"
)
label=vdf["UserRating"]

data=vdf.drop(columns=["UserRating","CreatedUTC","Title"])
data["UserName"]=data["UserName"].astype("string")
data["PostID"]=data["PostID"].astype("string")

history = slnn_model.fit(
    x=fed_csv,
    y="UserRating"
    epochs=5,
    batch_size=128,
    shuffle=False,
    random_seed=1234,
    dataset_builder=data_builder_dict,
    verbose=2,
)
# 细化情感分类函数
def sentiment_classification(texts):
    classifier = pipeline('sentiment-analysis', model=bert_model, tokenizer=tokenizer)
    results = classifier(texts)
    
    # 细化情感程度
    def refine_sentiment(score, label):
        if label == "NEGATIVE":
            if score > 0.9:
                return "Very Negative", score
            elif score > 0.7:
                return "Negative", score
            else:
                return "Slightly Negative", score
        elif label == "POSITIVE":
            if score > 0.9:
                return "Very Positive", score
            elif score > 0.7:
                return "Positive", score
            else:
                return "Slightly Positive", score
        else:
            return "Neutral", score

    refined_results = [refine_sentiment(res['score'], res['label']) for res in results]
    return refined_results
# 多次消极搜索提醒功能
def track_user_sentiment_history(user_input_history, max_negative_count=5):
    negative_count = sum(1 for sentiment in user_input_history if sentiment == "NEGATIVE" and score>0.7)
    if negative_count >= max_negative_count:
        print("您似乎多次搜索了消极信息，是否需要心理咨询等平台的帮助？")
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline

# 指定本地模型路径
model_name = '/home/quinn/scretflow/BERT'

# 加载BERT模型和分词器（本地路径），指定使用 PyTorch 权重
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertForSequenceClassification.from_pretrained(model_name, from_pt=True)

# 初始化情感分析管道
sentiment_analysis = pipeline("sentiment-analysis", model=bert_model, tokenizer=tokenizer)

# 用户输入历史
user_input_history = [
    "I'm terribly sad", 
    "Life is so hard", 
    "I feel down", 
    "I'm feeling really down and hopeless.", 
    "I'm a total loser"
]

# 获取用户情感分类
def get_sentiments(texts):
    return [sentiment_analysis(text)[0]['label'] for text in texts]

user_sentiments = get_sentiments(user_input_history)

# 记录用户情感历史
def track_user_sentiment_history(sentiments):
    global user_input_history
    user_input_history = sentiments

# 更新用户情感历史
track_user_sentiment_history(user_sentiments)

# 打印用户情感历史
print("User Sentiment History:", user_sentiments)
# 推荐帖子
def recommend_posts(user_input, posts_data):
    user_sentiment, _ = sentiment_classification([user_input])[0]

    if "Negative" in user_sentiment:
        positive_posts = posts_data[posts_data['sentiment_tags'].str.contains("Positive")]
        relevant_posts = pd.concat([positive_posts, posts_data])
    else:
        relevant_posts = posts_data

    # 根据BERT嵌入进行推荐（此处简化处理）
    recommended_posts = relevant_posts.head(5)
    return recommended_posts
user_input = "I feel stressed"
recommended_posts = recommend_posts(user_input, posts_data)
print("推荐的帖子：", recommended_posts)
#16
history
