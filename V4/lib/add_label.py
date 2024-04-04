import pickle
import numpy as np
import pandas as pd

# 组合userid和嵌入向量
user_combined_embeddings = []
item_combined_embeddings = []
for i in range(1, 6):
    result_file_path = f'datasets/data/partition/final/label_user_embeds_{i}.pk'
    data = pd.read_csv(file_path)
    # 读取CSV文件
    file_path = f'../datasets/data/partition/userlist_group_{i}.csv'
    data = pd.read_csv(file_path)

    # 提取第一列作为userid列表
    userid_list = data.iloc[:, 0].tolist()

    # 显示前几个userid验证结果
    print(userid_list[:5])


    # 加载嵌入向量文件
    with open(f'../datasets/data/partition/sub/subuser_embeds_{x}.pk', 'rb') as file:
        embeddings = pickle.load(file)
    with open(f'../datasets/data/partition/sub/subserv_embeds_{x}.pk', 'rb') as file:
        item_embeddings = pickle.load(file)
    for userid, embedding in zip(userid_list, embeddings):
        user_combined_embedding = np.insert(embedding, 0, userid)
        user_combined_embeddings.append(combined_embedding)
    item_combined_embeddings.append(item_embeddings)

    # # 打印前几个userid和它们的嵌入向量来验证
    # for i in range(5):  # 假设打印前5个进行检查
    #     print("UserID:", combined_embeddings[i][0], "Embedding:", combined_embeddings[i][1:])

#保存更新后的数据
with open(f'../datasets/data/partition/sub/updated_user_embeddings.pk', 'wb') as file:
    pickle.dump(user_combined_embeddings, file)
with open(f'../datasets/data/partition/sub/updated_item_embeddings.pk', 'wb') as file:
    pickle.dump(item_combined_embeddings, file)

# print("前几个用户嵌入向量的样本:",combined_embeddings[:5])
