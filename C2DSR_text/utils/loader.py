"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import codecs
import copy
import pdb
import pandas as pd

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.filename  = filename
        # ************* item_id *****************
        opt["source_item_num"] = self.read_item("./dataset/" + filename + "/Alist.txt")
        opt["target_item_num"] = self.read_item("./dataset/" + filename + "/Blist.txt")
        
        # ************* sequential data *****************


        source_train_data = "./dataset/" + filename + "/traindata_new.txt"
        source_valid_data = "./dataset/" + filename + "/validdata_new.txt"
        source_test_data = "./dataset/" + filename + "/testdata_new.txt"

        if self.opt["neg_type"] == "popular":
            self.target_negative_df = pd.read_csv("./dataset/" + self.filename + "/target_negative.csv",header=None)
            self.source_negative_df = pd.read_csv("./dataset/" + self.filename + "/source_negative.csv",header=None)
        

        if evaluation < 0:
            self.train_data = self.read_train_data(source_train_data)
            data = self.preprocess()
        elif evaluation == 2:
            self.test_data = self.read_test_data(source_valid_data)
            data = self.preprocess_for_predict()
        else :
            self.test_data = self.read_test_data(source_test_data)
            data = self.preprocess_for_predict()

        # shuffle for training
        if evaluation == -1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            #
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            # 为什么？
            # 因为在训练的时候，每个batch的数据都是随机的，所以最后一个batch的数据可能不足batch_size
            # 所以这里需要将最后一个batch的数据复制一份，使得最后一个batch的数据也能够达到batch_size
            # 这样就可以保证每个batch的数据都是batch_size
            # 但是这样做的话，最后一个batch的数据就会和前面的batch的数据重复，这样做的话，会不会影响模型的训练效果？
            # 不会，因为在训练的时候，每个batch的数据都是随机的，所以最后一个batch的数据和前面的batch的数据重复，不会影响模型的训练效果
            if len(data)%batch_size != 0:
                data += data[:batch_size]
                
            # 如果len(data)%batch_size = 0
            # 则取所有的数据
            data = data[: (len(data)//batch_size) * batch_size]
        else :
            batch_size = 2048
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

       
    def read_item(self, fname):
        item_number = 0
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            for line in fr:
                item_number += 1
        return item_number

    def read_train_data(self, train_file):
        def takeSecond(elem):
            return elem[1]
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            for id, line in enumerate(infile):
                res = []
            
                line = line.strip().split("\t")[2:]
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))
                # 使用时间戳进行排序，按照先后顺序排
                res.sort(key=takeSecond)
                res_2 = []
                for r in res:
                    res_2.append(r[0])
                train_data.append(res_2)

        return train_data

    def read_test_data(self, test_file):
        def takeSecond(elem):
            return elem[1]
        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            self.user_ids = []
            for id, line in enumerate(infile):
                res = []
                
                # 存储这个序列的User ID，帮助找到对应的负样本列表。
                self.user_ids.append(int(line.strip().split("\t")[0]))
                
                line = line.strip().split("\t")[2:]
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))

                # 没有必要进行排序了，因为原始的数据就是有序的。没有必要再多此一举。可能也会打断输入到模型中的物品顺序.因为不同领域时间是不是一样的.
                
                # 没有排序导致一个输出为空？TODO 对于原始数据需要排序？因为原始数据没有排好序！
                if self.opt["test_sort"]:
                   
                    res.sort(key=takeSecond)
                    

                # 取数据,一直到最后一个数据.
                res_2 = []
                for r in res[:-1]:
                    # 取物品ID，
                    res_2.append(r[0])
                    
                # 判断最后一个物品是属于哪个领域物品
                if res[-1][0] >= self.opt["source_item_num"]: # denoted the corresponding validation/test entry
                    # 如果是目标领域的物品，则是1，否则是0
                    test_data.append([res_2, 1, res[-1][0]])
                else :
                    test_data.append([res_2, 0, res[-1][0]])
                    
        return test_data

    def preprocess_for_predict(self):

        # 如果 是 HVIDIO的数据集话，需要设置最大长度为30
        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            # for origin C2DSR dataset 
            # max_len = 15
            # self.opt["maxlen"] = 15
            
            # for MGCL dataset length
            max_len = self.opt["maxlen"]
        print("max_len: ",max_len)
        print("the negative type is: ",self.opt["neg_type"])
            
        processed=[]
        index = 0
       
        for d in self.test_data: # the pad is needed! but to be careful.
             # d的结构是： sequence， 是否是目标领域的物品， 最后一个物品的ID
            # [list,大int,int]
            # 截断数据，避免超出最长度，前面最多max_len-1个物品，因为最后一个物品被提取出来了。
            d[0] = d[0][-(max_len-1):]
            position = list(range(len(d[0])+1))[1:] # 生成一个序列的位置，从1开始

            xd = []
            xcnt = 1
            x_position = []

            yd = []
            ycnt = 1
            y_position = []

            for w in d[0]:
               
                # 取物品ID
                # 判断是哪个领域的物品
                # postion例子表示
                # x领域有值，y领域置0
                if w < self.opt["source_item_num"]:
                    xd.append(w)
                    x_position.append(xcnt)
                    xcnt += 1
                    yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    y_position.append(0)


                else:
                    xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    x_position.append(0)
                    yd.append(w)
                    y_position.append(ycnt)
                    ycnt += 1


            if len(d[0]) < max_len:
                # 补全空序列位置
                # 为什么之前的数据加载前面，因为预测是预测后面的，参考sasrec，要将前面的位置设置为空。
                position = [0] * (max_len - len(d[0])) + position
                x_position = [0] * (max_len - len(d[0])) + x_position
                y_position = [0] * (max_len - len(d[0])) + y_position

                xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d[0])) + xd
                yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d[0])) + yd
                seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(d[0])) + d[0]

            # 这段代码表示：判断最后一个有效的位置是哪个
            
            x_last = -1
            for id in range(len(x_position)):
                id += 1
                if x_position[-id]:
                    x_last = -id
                    break

            y_last = -1
            for id in range(len(y_position)):
                id += 1
                if y_position[-id]:
                    y_last = -id
                    break

            negative_sample = []
            
            if self.opt["neg_type"] == "popular":
                
                # 根据序列第一个位置获取用户的ID，然后根据用户的ID获取负样本
                uid = self.user_ids[index]
                index +=1
                # 直接从文件中读取负样本集合，已经对负样本进行了处理，因此不会和正样本重复。
                if d[1]: # target domain:
                
                    # 获取用户的负样本
                    negative_sample = self.target_negative_df[self.target_negative_df[0]==uid].values.tolist()[0][1:]
                else:# source domain:
                    
                    negative_sample = self.source_negative_df[self.source_negative_df[0]==uid].values.tolist()[0][1:]
            elif self.opt["neg_type"] == "random":
                
                for i in range(999):
                    # 总共是999个负样本，根据预测的物品是属于哪个领域，生成哪个领域的负样本
                    while True:
                        if d[1] : # in Y domain, the validation/test negative samples
                        # 从 0 开始映射，并没有给所有的映射！。
                            sample = random.randint(0, self.opt["target_item_num"] - 1)
                            if sample != d[2] - self.opt["source_item_num"]:
                                negative_sample.append(sample)
                                break
                        else : # in X domain, the validation/test negative samples
                            sample = random.randint(0, self.opt["source_item_num"] - 1)
                            if sample != d[2]:
                                negative_sample.append(sample)
                                break

            
            
            if d[1]:
                processed.append([seq, xd, yd, position, x_position, y_position, x_last, y_last, d[1], d[2]-self.opt["source_item_num"], negative_sample])
            else:
                processed.append([seq, xd, yd, position, x_position, y_position, x_last, y_last, d[1],
                                  d[2], negative_sample])
            
        return processed

    def preprocess(self):

        def myprint(a):
            for i in a:
                print("%6d" % i, end="")
            print("")
        """ Preprocess the data and convert to ids. """
        processed = []


        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            # for origin C2DSR dataset 
            # max_len = 15
            # self.opt["maxlen"] = 15
            
            # for MGCL dataset length
            max_len = self.opt["maxlen"] 
        print("train_max_len: ",max_len)

        for d in self.train_data: # the pad is needed! but to be careful.
            # 拿到单个用户的交互数据
            
            # 注意超过最大长度的序列，需要从后往前进行截断
            d = d[-max_len:]
            ground = copy.deepcopy(d)[1:]


            share_x_ground = []
            share_x_ground_mask = []
            share_y_ground = []
            share_y_ground_mask = []
            # 遍历所有的物品数据
            for w in ground:
                # 判断如果物品的ID小于这个源目标域的最大数，则是这个领域的东西
                if w < self.opt["source_item_num"]:
                    share_x_ground.append(w)
                    share_x_ground_mask.append(1)
                    share_y_ground.append(self.opt["target_item_num"])
                    share_y_ground_mask.append(0)
                # 否则是另外一个领域的
                else:
                    share_x_ground.append(self.opt["source_item_num"])
                    share_x_ground_mask.append(0)
                    share_y_ground.append(w - self.opt["source_item_num"])
                    share_y_ground_mask.append(1)

            # d 是混合序列
            # 去除这个用户的最后一个物品
            # 这个物品是用来做验证的
            d = d[:-1]  # delete the ground truth
            # 生成一个序列的位置
            position = list(range(len(d)+1))[1:]
            ground_mask = [1] * len(d)

            
            # 含义是：如果是源领域的物品，则是物品的ID，否则是源领域的物品的数量
            xd = []
            xcnt = 1
            # 含义是：如果是源领域的物品，则是物品的位置，否则是0
            x_position = []


            yd = []
            ycnt = 1
            y_position = []

            #corru_x 表示
            corru_x = []
            corru_y = []
            # 下面是获取 corrupted sequence ，是用来做负样本的
            # 即 在domain x的序列中 PAD的位置 使用随机生成Y领域的数据来替代了，而不是0向量了。
            for w in d:
                if w < self.opt["source_item_num"]:
                    corru_x.append(w)
                    xd.append(w)
                    x_position.append(xcnt)
                    xcnt += 1
                    
                    corru_y.append(random.randint(0, self.opt["source_item_num"] - 1))
                    yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    y_position.append(0)

                else:
                    corru_x.append(random.randint(self.opt["source_item_num"], self.opt["source_item_num"] + self.opt["target_item_num"] - 1))
                    xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    x_position.append(0)
                    corru_y.append(w)
                    yd.append(w)
                    y_position.append(ycnt)
                    ycnt += 1

            now = -1
            x_ground = [self.opt["source_item_num"]] * len(xd) # caution!
            x_ground_mask = [0] * len(xd)
            for id in range(len(xd)):
                id+=1
                if x_position[-id]:
                    if now == -1:
                        now = xd[-id]
                        if ground[-1] < self.opt["source_item_num"]:
                            x_ground[-id] = ground[-1]
                            x_ground_mask[-id] = 1
                        else:
                            xd[-id] = self.opt["source_item_num"] + self.opt["target_item_num"]
                            x_position[-id] = 0
                    else:
                        x_ground[-id] = now
                        x_ground_mask[-id] = 1
                        now = xd[-id]
            if sum(x_ground_mask) == 0:
                # 运行到这的含义表示为，这条序列中不存在target 领域的物品
                print("pass sequence x")
                continue

            now = -1
            y_ground = [self.opt["target_item_num"]] * len(yd) # caution!
            y_ground_mask = [0] * len(yd)
            for id in range(len(yd)):
                id+=1
                if y_position[-id]:
                    if now == -1:
                        now = yd[-id] - self.opt["source_item_num"]
                        if ground[-1] > self.opt["source_item_num"]:
                            y_ground[-id] = ground[-1] - self.opt["source_item_num"]
                            y_ground_mask[-id] = 1
                        else:
                            yd[-id] = self.opt["source_item_num"] + self.opt["target_item_num"]
                            y_position[-id] = 0
                    else:
                        y_ground[-id] = now
                        y_ground_mask[-id] = 1
                        now = yd[-id] - self.opt["source_item_num"]
            if sum(y_ground_mask) == 0:
                print("pass sequence y")
                continue

            if len(d) < max_len:
                position = [0] * (max_len - len(d)) + position
                x_position = [0] * (max_len - len(d)) + x_position
                y_position = [0] * (max_len - len(d)) + y_position

                ground = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + ground
                share_x_ground = [self.opt["source_item_num"]] * (max_len - len(d)) + share_x_ground
                share_y_ground = [self.opt["target_item_num"]] * (max_len - len(d)) + share_y_ground
                x_ground = [self.opt["source_item_num"]] * (max_len - len(d)) + x_ground
                y_ground = [self.opt["target_item_num"]] * (max_len - len(d)) + y_ground

                ground_mask = [0] * (max_len - len(d)) + ground_mask
                share_x_ground_mask = [0] * (max_len - len(d)) + share_x_ground_mask
                share_y_ground_mask = [0] * (max_len - len(d)) + share_y_ground_mask
                x_ground_mask = [0] * (max_len - len(d)) + x_ground_mask
                y_ground_mask = [0] * (max_len - len(d)) + y_ground_mask

                corru_x = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + corru_x
                corru_y = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + corru_y
                xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + xd
                yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + yd
                d = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + d
            else:
                print("pass, len: ",len(d))
            
            if len(d)>max_len:
                print(len(d),len(xd),len(yd),len(position),len(x_position),len(y_position),len(ground),len(share_x_ground),len(share_y_ground),len(x_ground),len(y_ground),len(ground_mask),len(share_x_ground_mask),len(share_y_ground_mask),len(x_ground_mask),len(y_ground_mask),len(corru_x),len(corru_y))
            
            processed.append([d, xd, yd, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y])
            
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval!=-1:
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]))
        else :
            batch = list(zip(*batch))

            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]), torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]))

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


