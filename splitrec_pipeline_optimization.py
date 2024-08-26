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
f=open("/home/sherlock/Documents/post/ratings.dat","r",encoding="unicode_escape")

users_data=load_data(
    "/home/sherlock/Documents/post/users.dat",
    columns=["UserName","Gender","Age"],
)

posts_data=load_data("/home/sherlock/Documents/post/posts.dat",columns=["PostID","Title","NumComments"])
ratings_columns=["UserName","PostID","UserRating","CreatedUTC"]

rating_data=load_data("/home/sherlock/Documents/post/ratings.dat",columns=ratings_columns)

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

#10
from secretflow.ml.nn.applications.sl_deep_fm import DeepFMbase,DeepFMfuse
from secretflow.ml.nn import SLModel

NUM_USERS=197
NUM_POSTS=631
GENDER_VOCAB=["Female","Male","Other","Non-binary"]
AGE_VOCAB=[1,18,25,35,45,50,56]
COMMENT_VOCAB=[1,1000,4000,8000,10000,15000]

#11
def create_base_model_alice():
    def create_model():
        import tensorflow as tf

        def preprocess():
            inputs={
                "UserName":tf.keras.Input(shape=(1,),dtype=tf.string),
                "Gender":tf.keras.Input(shape=(1,),dtype=tf.string),
                "Age":tf.keras.Input(shape=(1,),dtype=tf.int64),
            }
            user_id_output=tf.keras.layers.Hashing(
                num_bins=NUM_USERS,output_mode="one_hot"
            )
            user_gender_output=tf.keras.layers.StringLookup(
                vocabulary=GENDER_VOCAB,output_mode="one_hot"
            )

            user_age_out=tf.keras.layers.IntegerLookup(
                vocabulary=AGE_VOCAB,output_mode="one_hot"
            )


            outputs={
                "UserName":user_id_output(inputs["UserName"]),
                "Gender":user_gender_output(inputs["Gender"]),
                "Age":user_age_out(inputs["Age"]),
            }
            return tf.keras.Model(inputs=inputs,outputs=outputs)

        preprocess_layer=preprocess()
        model=DeepFMbase(
            dnn_units_size=[256,32],
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
            }
            post_id_out = tf.keras.layers.Hashing(
                num_bins=NUM_POSTS, output_mode="one_hot"
            )

            post_comment_output = tf.keras.layers.IntegerLookup(
                vocabulary=COMMENT_VOCAB, output_mode="one_hot"
            )

            outputs = {
                "PostID": post_id_out(inputs["PostID"]),
                "NumComments": post_comment_output(inputs["NumComments"]),
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

#13
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

#14
base_model_dict={alice:create_base_model_alice(),bob:create_base_model_bob()}
model_fuse=create_fuse_model()

#15
from secretflow.data.vertical import read_csv as v_read_csv
from secretflow.ml.nn import SLModel

vdf=v_read_csv(
    {alice:"alice_reddit.csv",bob:"bob_reddit.csv"},keys="ID",drop_keys="ID"
)
label=vdf["UserRating"]

data=vdf.drop(columns=["UserRating","CreatedUTC","Title"])
data["UserName"]=data["UserName"].astype("string")
data["PostID"]=data["PostID"].astype("string")

sl_model_origin=SLModel(
    base_model_dict=base_model_dict,
    device_y=bob,
    model_fuse=model_fuse,
)

sl_model_origin = SLModel(
    base_model_dict=base_model_dict,
    device_y=bob,
    model_fuse=model_fuse,
)


sl_model_pipeline = SLModel(
    base_model_dict=base_model_dict,
    device_y=bob,
    model_fuse=model_fuse,
    strategy='pipeline',
    pipeline_size=2,
)

import time

histories = []
cost_time = []
for sl_model in [sl_model_origin, sl_model_pipeline]:
    begin = time.time()
    history = sl_model.fit(
        data,
        label,
        validation_data=(data, label),
        epochs=3,
        batch_size=32,
        shuffle=True,
        verbose=1,
        validation_freq=1,
    )
    end = time.time()
    cost_time.append((end - begin) / 60)
    histories.append(history)

print(cost_time)