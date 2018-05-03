# LFM
import pandas as pd
import datetime
import math
import random

# 读取数据
def read_data(path):
    f = open(path, 'r', encoding='utf-8')
    data = pd.DataFrame(columns=('userid', 'itemid', 'rate')) # 用DataFrame的好处是索引和key格式自由
    cnt = 0
    while 1:
        line = f.readline() #逐行读取
        if not line:
            break
        line = line.split()
        data.loc[cnt] = line[:2] + [int(line[2])]
        cnt += 1
    print('build over')
    return data

def LFM(train, F, n): # train是传入数据， F是类数， n是迭代次数
    user_matrix = dict()
    item_matrix = dict()
    alpha = 0.01
    lamda = 0.02
    for index in range(train.count()[0]):
        user, item, rating = list(train.values[index])
        if user not in user_matrix: # 随机初始化
            user_matrix[user] = [random.random() / math.sqrt(F) for f in range(F)]
        if item not in item_matrix:
            item_matrix[item] = [random.random() / math.sqrt(F) for f in range(F)]

    for step in range(n): # 迭代n步
        for index in range(train.count()[0]):
            user, item, rui = list(train.values[index]) # 用户， 物品， 评分
            rui_hat = predict(user, item, user_matrix, item_matrix)
            eui = rui - rui_hat # 根据当前参数计算误差
            for f in range(F): # 迭代公式 F分类
                user_matrix[user][f] += alpha * (eui * item_matrix[item][f] - lamda * user_matrix[user][f])
                item_matrix[item][f] += alpha * (eui * user_matrix[user][f] - lamda * item_matrix[item][f])
        alpha *= 0.9
    return user_matrix, item_matrix

# 读取用户矩阵和物品矩阵，求得预测值
def predict(user, item, user_matrix, item_matrix):
    sum = 0.0
    for f in range(len(user_matrix[user])):
        sum += user_matrix[user][f] * item_matrix[item][f]
    return sum

# 写入predict.dat
def write_prediction(test_path,predict_path,user_matrix,item_matrix):
    # 打开文件
    test_file = open(test_path, 'r', encoding='utf-8')
    predict_file = open(predict_path, 'w', encoding='utf-8', newline='')
    while 1:
        line = test_file.readline()
        if not line:
            break
        line=line.split('\n')
        line_data = line[0].split()
        if not (line_data[0] in user_matrix and line_data[1] in item_matrix):
            continue
        p_rate = predict(line_data[0], line_data[1], user_matrix, item_matrix)
        predict_file.write(line[0] + ' ' + str(round(p_rate))+'\n') # 将结果写入

if __name__ ==  '__main__':
    starttime = datetime.datetime.now() #计时

    train_path = './data/train.dat'
    test_path = './data/test.dat'
    predict_path = './data/predict.dat'
    data = read_data(train_path)
    user_matrix, item_matrix = LFM(data, 100, 100)
    write_prediction(test_path,predict_path,user_matrix,item_matrix)

    endtime = datetime.datetime.now()
    print(endtime - starttime)
