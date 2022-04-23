import pandas as pd
import math

def load_dataset(dataset,type,min_length):
    if dataset == 'ASSISTment2009':
        path = "./Datasets/" + dataset + "/raw.csv"
        usecols = ['order_id', 'user_id', 'problem_id', 'correct', 'skill_id', 'type']
        print('Loading dataset from', path, ' with cols:', usecols)
        csv_data = pd.read_csv(path, usecols=usecols)
        # 按时间顺序对数据排序
        csv_data.sort_values(['order_id'], ascending=True)
        # 选择特定类型（type）的数据
        print('Choosing data where type=', type)
        type_data = csv_data[csv_data['type'] == type]
        user_list = type_data['user_id'].tolist()
        problem_list = type_data['problem_id'].tolist()
        correct_list = type_data['correct'].tolist()
        skill_list = type_data['skill_id'].tolist()
        # 每一行为[user_id，problem_id，correct，skill_id]
        data_raw = [user_list, problem_list, correct_list, skill_list]
    elif dataset == 'AICFE':
        path = "./Datasets/" + dataset + "/"+type+"/unit-"+type+".csv"
        file = open(path)
        lines = file.readlines()
        skip = 0
        user_list = []
        problem_list = []
        correct_list = []
        skill_list = []
        for line in lines:
            if skip != 0:
                data = line.strip('\n').split(',')
                user = data[0]
                problem = data[3]
                skill = data[4]
                score = data[5]
                full_score = data[6]
                if skill != 'n.a.' and full_score != 'n.a.':
                    user_list.append(user)
                    problem_list.append(problem)
                    skill_list.append(skill)
                    if score == full_score:
                        correct_list.append(1)
                    else:
                        correct_list.append(0)
            else:
                skip += 1
        data_raw = [user_list, problem_list, correct_list, skill_list]

    #统计用户序列长度
    user_count = {}
    for i in range(data_raw[0].__len__()):
        userid = data_raw[0][i]
        if user_count.__contains__(userid):
            user_count[userid] += 1
        else:
            user_count[userid] = 1
    #过滤与重新编号
    user_id = {}
    item_id = {}
    skill_id = {}
    user_list_filtered = []
    item_list_filtered = []
    correct_list_filtered = []
    filtered_Q_matrix = []
    print('Filting data where sequence length>=',min_length)
    for i in range(data_raw[0].__len__()):
        user = data_raw[0][i]
        item = data_raw[1][i]
        correct = data_raw[2][i]
        if dataset == 'ASSISTment2009':
            skillids = data_raw[3][i].split(',')
        else:
            skillids = data_raw[3][i].split('~~')

        if user_count[user] >= min_length:
            if not user_id.__contains__(user):
                user_id[user] = user_id.__len__()
            if not item_id.__contains__(item):
                item_id[item] = item_id.__len__()
                skills = []
                for skill in skillids:
                    if not skill_id.__contains__(skill):
                        skill_id[skill] = skill_id.__len__()
                    skills.append(skill_id[skill])
                filtered_Q_matrix.append(skills)
            user_list_filtered.append(user_id[user])
            item_list_filtered.append(item_id[item])
            correct_list_filtered.append(correct)
    print(user_id)
    print(skill_id)
    return [user_list_filtered,item_list_filtered,correct_list_filtered,filtered_Q_matrix]


def get_split_triplet(dataset, type, min_length):
    [user_list, item_list, correct_list,Q_matrix] = load_dataset(dataset, type, min_length)
    user_num = max(user_list) + 1
    item_num = max(item_list) + 1
    skill_num = max([max(i) for i in Q_matrix]) + 1
    record_num = user_list.__len__()
    # all_sequences = {userid:[[itemids,...],[correct,...]]}
    all_sequences = {}
    for i in range(user_list.__len__()):
        if all_sequences.__contains__(user_list[i]):
            all_sequences[user_list[i]][0].append(item_list[i])
            all_sequences[user_list[i]][1].append(correct_list[i])
        else:
            all_sequences[user_list[i]] = [[item_list[i]],[correct_list[i]]]
    # train_triplet [[userid,itemid,corect],...]
    # test_triplet [[userid,itemid,corect],...]
    train_triplet = []
    test_triplet = []
    for user in all_sequences:
        sequence_length = all_sequences[user][0].__len__()
        train_length = sequence_length-1
        for index in range(sequence_length):
            if index < train_length:
                train_triplet.append([user,
                                      all_sequences[user][0][index],
                                      all_sequences[user][1][index]])
            else:
                test_triplet.append([user,
                                      all_sequences[user][0][index],
                                      all_sequences[user][1][index]])

    return user_num,item_num,skill_num,record_num,train_triplet,test_triplet,Q_matrix


def get_split_sequences(dataset, type, min_length):
    [user_list, item_list, correct_list, Q_matrix] = load_dataset(dataset, type, min_length)
    user_num = max(user_list) + 1
    item_num = max(item_list) + 1
    skill_num = max([max(i) for i in Q_matrix]) + 1
    record_num = user_list.__len__()
    # all_sequences = {userid:[[itemids,...],[correct,...]]}
    all_sequences = {}
    for i in range(user_list.__len__()):
        if all_sequences.__contains__(user_list[i]):
            all_sequences[user_list[i]][0].append(item_list[i])
            all_sequences[user_list[i]][1].append(correct_list[i])
        else:
            all_sequences[user_list[i]] = [[item_list[i]], [correct_list[i]]]

    train_sequences = {}
    test_triplet = []
    # train_sequences = {userid:[[itemids,...],[correct,...]]}
    # test_triplet [[userid,itemid,corect],...]
    for user in all_sequences:
        sequence_length = all_sequences[user][0].__len__()
        train_length = sequence_length - 1
        train_sequences[user] = [[all_sequences[user][0][0:train_length]],
                                 [all_sequences[user][1][0:train_length]]]
        test_item_sequence = all_sequences[user][0][train_length:]
        test_correct_sequence = all_sequences[user][1][train_length:]
        for i in range(test_item_sequence.__len__()):
            test_triplet.append([user,test_item_sequence[i],test_correct_sequence[i]])
    return user_num,item_num,skill_num,record_num,train_sequences,test_triplet,Q_matrix



if __name__ == '__main__':
    # dataset = 'ASSISTment2009'
    # type = 'RandomIterateSection'
    dataset = 'AICFE'
    type = 'math'
    min_length = 10
    load_dataset(dataset, type, min_length)

