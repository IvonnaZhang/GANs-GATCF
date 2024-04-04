import pickle as pk

import numpy as np

## Read as bytes
# with open('./datasets/rtMatrix.txt', 'rb') as f:
#     data = f.read()
#
# # Save as pickle
# with open('data.pk', 'wb') as f:
#     pk.dump(data, f)

#pk.dump(data, open(f'data.pk', 'wb'))

# Load as pickle
with open('./datasets/data/partition/RecEarser_5.pk', 'rb') as f:
    data_pickle = pk.load(f)
    f = np.array(data_pickle)

    print(f.shape)
    # print(f.dtype)
    print("第一组:", f[0])
    print("第二组:", f[1])
    print("第三组:", f[2])
    print("第四组:", f[3])
    print("第五组:", f[4])

keys_list = []
for i in range(5):
    group = data_pickle[i]
    if isinstance(group, dict):
        group_keys = list(group.keys())
        keys_list.append(group_keys)
    else:
        print(f"Element at index {i} is not a dictionary.")

for i, keys in enumerate(keys_list):
    print(f"第{i+1}组 keys:", keys)