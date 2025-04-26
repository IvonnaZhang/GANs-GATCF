import csv
import pickle as pk

def read_csv_file(filepath):
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)

def read_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        return pk.load(f)

def write_filtered_rows(filename, data_list, keys):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key in keys:
            row_number = key+1  #这里为什么要加一
            if 0 <= row_number < len(data_list):
                writer.writerow(data_list[row_number])
            else:
                print(f"Row number {row_number} is out of bounds for {filename}.")

def packet(userlist_path, wslist_path, data_pickle_path):

    userlist = read_csv_file(userlist_path)
    wslist = read_csv_file(wslist_path)

    # Load the pickle file containing groups and keys
    data_pickle = read_pickle_file(data_pickle_path)

    # Extract keys for different groups
    keys_list = []
    for i, group in enumerate(data_pickle):
        if isinstance(group, dict):
            keys_list.append(list(group.keys()))
        else:
            print(f"Element at index {i} is not a dictionary.")

    # Filter and write userlist and wslist based on keys
    for i, keys in enumerate(keys_list):
        userlist_filename = f'./datasets/data/partition/userlist_group_{i+1}.csv'
        wslist_filename = f'./datasets/data/partition/wslist_group_{i+1}.csv'
        write_filtered_rows(userlist_filename, userlist, keys)
        write_filtered_rows(wslist_filename, wslist, keys)

