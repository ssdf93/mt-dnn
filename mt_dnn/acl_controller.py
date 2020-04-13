import sys
import json
import torch
import random
import copy
import numpy as np
from shutil import copyfile


class ACLController():
    def __init__(self, n_task, max_cnt=3, batch_size=None, phi=0.5, max_step=1000):
        self.n_task = n_task
        self.data_loss_list = [myQueue(max_cnt, batch_size=batch_size) for _ in range(n_task)]
        self.task_index = -1
        self.phi = phi
        self.cur_step = 0
        self.max_step = max_step
    
    def initalization(self, phi=None, max_step=None):
        if max_step:
            self.max_step = max_step
        if phi:
            self.phi = phi
        self.cur_step = 0
        for data in self.data_loss_list:
            data.empty()

    def insert(self, task_id, data, loss):
        self.data_loss_list[task_id].append((data, loss))
    
    def calculate_loss(self):
        losses = []
        for i, data in enumerate(self.data_loss_list):
            loss = data.calculate_loss()
            losses.append(loss)
        return losses
    
    def step(self, model):
        if self.task_index == self.n_task - 1:
            losses = self.calculate_loss()
            # print(losses)
            if random.random() < self.phi:
                arg_task_index = np.argmax(losses)
                # print("Choose max index, ", arg_task_index)
            else:
                p = np.array(losses)
                p /= p.sum()
                arg_task_index = np.random.choice(list(range(self.n_task)), p=p, replace=False)
                # print("Choose random index, ", arg_task_index)
            # update task
            for data in self.data_loss_list[arg_task_index]:
                batch_meta, batch_data = data
                # print("batch_meta size", batch_meta)
                # print("batch_data size", len(batch_data))
                # for tem in batch_data:
                #     print(tem.size())
                model.update(batch_meta, batch_data)
            self.data_loss_list[arg_task_index].empty()

    def get_task_id(self):
        self.task_index = (self.task_index+1) % self.n_task
        self.cur_step += 1
        if self.cur_step > self.max_step:
            return None
        return self.task_index
        

class myQueue():
    def __init__(self, n=3, batch_size=None):
        self.n = n
        self.data = []
        self.batch_size = batch_size

    def append(self, item):
        if len(self.data) >= self.n:
            self.data.pop(0)
        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def empty(self):
        self.data = []
    
    def calculate_loss(self):
        sum_loss = 0.0
        cnt = len(self.data)
        for i, (data,loss) in enumerate(self.data):
            sum_loss += loss
        if cnt > 0:
            sum_loss /= cnt
        return sum_loss
    
    def __iter__(self): 
        total_meta = None
        total_data = None
        uids = []
        if self.batch_size:
            for i, (data, loss) in enumerate(self.data):
                if i == 0:
                    total_meta, total_data = data
                    uids = copy.deepcopy(total_meta['uids'])
                else:
                    batch_meta, batch_data = data
                    uids += batch_meta['uids']
                    print(total_meta)
                    print(batch_meta)
                    print(len(total_data))
                    print(len(batch_data))
                    print(total_data)
                    print(batch_data)
                    for j,(total, new_data) in enumerate(zip(total_data, batch_data)):
                        print(total.size(), new_data.size())
                        total_data[j] = torch.cat([total, new_data], dim=0)
            total_len = len(uids)
            for tmp in total_data:
                print(tmp.size())
            for i in range(0, total_len, self.batch_size):
                batch_uids = {"uids" : uids[i:i+self.batch_size]}
                total_meta.update(batch_uids)
                new_batch_data = [data[i:i+self.batch_size] for data in total_data]
                yield (total_meta, new_batch_data)
        else:
            for i, (data, loss) in enumerate(self.data):
                yield data
