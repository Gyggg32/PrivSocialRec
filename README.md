# PrivSocialRec
##### 本项目主要是实现了一个基于隐语平台的信息筛选推荐系统，并与情感分析结合，对于长期情绪消极的用户及时与平台反馈，达到关注并维护用户心理健康的目的。
## 具体工作
##### 引入情感驱动的个性化推荐机制，利用自然语言处理技术进行情感分析，根据用户情绪状态动态调整推荐内容。
##### 结合 DeepFM 算法和 BERT 模型，通过因子分解机和深度神经网络捕捉用户与内容的特征交互，提升推荐的精准度。
##### 实现自学习模型，根据用户行为和反馈自适应调整推荐策略，确保推荐内容始终贴近用户的最新兴趣
## 项目创新点
##### 采用拆分学习模型，结合隐语平台的垂直联邦学习技术，实现跨域数据的分布式训练，保护用户隐私。
##### 与情感驱动模型相结合，能够实时分析用户情感状态。
##### 破除信息茧房的功能，引导用户探索未关注的帖子和子话题。
## 项目代码说明
### Emotion-Driven Recommendation .py
##### 实现情感驱动模型的核心代码，用户在不需要主动向外界求助心理疏导的情况下，以平台作为心理咨询师的形式隐形的对用户进行情绪疏导，用户视角来看无外界干预。
### splitrec.py
#####
### splitrec_pipeline_optimization.py
##### 该段代码实现了流水线并行，大大提高了推荐系统的速度。
### splitrec_qfp_optimization.py
##### 
