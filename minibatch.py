#coding=utf-8
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import sys
floder='/home/wujj/Code/Terrorist_Pattern/dgrec_final/'
sys.path.append(floder)
from neigh_samplers import UniformNeighborSampler
from utils import *
from sklearn.preprocessing import MinMaxScaler


class MinibatchIterator(object):
    
    def __init__(self, 
                adj_info, # in pandas dataframe
                item_id_map,
                latest_sessions,
                data, # data list, either [train, valid] or [train, valid, test].
                placeholders,
                batch_size,
                max_degree,
                num_nodes,
                max_length,
                friend_layer_num,
                samples,
                dims,
                training=True):
        with open('/data/Experiment/random_seed.txt','r') as f:
            seed=int(f.read())
        np.random.seed(seed)
        self.num_layers = friend_layer_num
        self.adj_info = adj_info
        self.latest_sessions = latest_sessions
        self.training = training
        self.train_df, self.valid_df, self.test_df,df = data
        self.all_data = pd.concat([self.train_df, self.valid_df, self.test_df])
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.num_nodes = num_nodes
        self.max_length = max_length
        self.samples = samples
        self.dims = dims
        self.visible_time = self.user_visible_time()
        self.test_adj, self.test_deg = self.construct_test_adj()
        self.sizes=[1]
        
        day_limit=pd.Timestamp(2014,11,1)
        # day_limit=pd.Timestamp(2019,1,1)
        # day_limit=pd.Timestamp(2010,1,1)
        timestamp_limit=pd.Timestamp.timestamp(day_limit)
        temp_df1=self.all_data.groupby('UserId')['Timestamp'].max()
        temp_df1=self.all_data[self.all_data['UserId'].isin(temp_df1[temp_df1>=timestamp_limit].index.values)]
        temp_df2=temp_df1.groupby('UserId')['TimeId'].max()
        predict_session_ids=[]

        UserIds=temp_df2.index.values
        TimeIds=temp_df2.values
        for i in range(len(temp_df2)):
            predict_session_ids.append('%s_%s'%(int(UserIds[i]),int(TimeIds[i])))
        self.predict_df=self.all_data[self.all_data['SessionId'].isin(predict_session_ids)].reset_index(drop=True)

        n=1
        for i in range(self.num_layers):
            k=self.num_layers-i-1
            n*=self.samples[k]
            self.sizes.append(n)

        cols=df.columns.values[2:]
        new_id=[item_id_map[i] for i in item_id_map.keys()]
        item_id_df= pd.DataFrame({'ItemId':item_id_map.keys(),'Id':new_id})
        item_id_df['Id']=item_id_df['Id'].astype(np.int64)
        grid_id_df=self.all_data[['ItemId','Grid']].drop_duplicates()
        item_grid_df=pd.merge(item_id_df,grid_id_df, how='left',left_on=['Id'],right_on=['ItemId'],suffixes=('', '_'))
        index=df['ItemId'].isin(item_grid_df['Grid'].unique())
        df_final=pd.merge(df[index],item_grid_df, how='left',left_on=['ItemId'],right_on=['Grid'],suffixes=('', '__s'))
        df_final=df_final.sort_values(by=['Id','TimeId'],ascending=True).reset_index(drop=True)
        X=df_final[cols].values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        df_final[cols]=X_scaled
        self.dicts={}
        for timeid in df_final['TimeId'].unique():
            temp=df_final[df_final['TimeId']==timeid]
            self.dicts[str(int(timeid))]=temp[cols].values
        if self.training:
            self.adj, self.deg = self.construct_adj()
            self.train_session_ids,self.train_num_sessions,self.train_num_data= self._remove_infoless(self.train_df, self.adj, self.deg)
            self.valid_session_ids,self.valid_num_sessions,self.valid_num_data = self._remove_infoless(self.valid_df, self.test_adj,
                                                                                                       self.test_deg)
            self.sampler = UniformNeighborSampler(self.adj, self.visible_time, self.deg)
        self.test_session_ids,self.test_num_sessions,self.test_num_data  = self._remove_infoless(self.test_df, self.test_adj, self.test_deg)
       
        self.padded_data, self.mask = self._padding_sessions(self.all_data)
        self.test_sampler = UniformNeighborSampler(self.test_adj, self.visible_time, self.test_deg)
        
        self.batch_num = 0
        self.batch_num_val = 0
        self.batch_num_test = 0
        self.batch_num_predict = 0
        self.predict_session_ids,self.predict_num_sessions,self.predict_num_data  = self._remove_infoless(self.predict_df, self.test_adj, self.test_deg)

    def renew_predict_session_ids(self,day_limit1,timestamp_limit2):
        timestamp_limit1=pd.Timestamp.timestamp(day_limit1)
        temp_df1=self.all_data.groupby('UserId')['Timestamp'].max()
        temp_df1=self.all_data[self.all_data['UserId'].isin(temp_df1[(temp_df1>=timestamp_limit1)&(temp_df1<timestamp_limit2)].index.values)]
        temp_df2=temp_df1.groupby('UserId')['TimeId'].max()
        predict_session_ids=[]
        UserIds=temp_df2.index.values
        TimeIds=temp_df2.values
        for i in range(len(temp_df2)):
            predict_session_ids.append('%s_%s'%(int(UserIds[i]),int(TimeIds[i])))
        self.predict_df=self.all_data[self.all_data['SessionId'].isin(predict_session_ids)].reset_index(drop=True)
        self.predict_session_ids,self.predict_num_sessions,self.predict_num_data  = self._remove_infoless(self.predict_df, self.test_adj, self.test_deg)

        

    def user_visible_time(self):
        '''
            Find out when each user is 'visible' to her friends, i.e., every user's first click/watching time.
        '''
        visible_time = []
        for l in self.latest_sessions:
            timeid = max(loc for loc, val in enumerate(l) if val == 'NULL') + 1
            visible_time.append(timeid)
            assert timeid > 0 and timeid < len(l), 'Wrong when create visible time {}'.format(timeid)
        return visible_time

    def _remove_infoless(self, data, adj, deg):
        '''
        Remove users who have no sufficient friends.
        '''
        data = data.loc[deg[data['UserId']] != 0]
        reserved_session_ids = []
        for sessid in data.SessionId.unique():
            userid, timeid = sessid.split('_')
            userid, timeid = int(userid), int(timeid)
            flag = 0
            if self.num_layers==1:
                for neighbor in adj[userid, : ]:
                    if self.visible_time[neighbor] <= timeid:
                        flag=1
                        break
            elif self.num_layers==2:
                for neighbor in adj[userid, : ]:
                    if self.visible_time[neighbor] <= timeid and deg[neighbor] > 0:
                        for second_neighbor in adj[neighbor, : ]:
                            if self.visible_time[second_neighbor] <= timeid:
                                flag=1
                                break
                    if flag==1:
                        break
            elif self.num_layers==3:
                for neighbor in adj[userid, : ]:
                    if self.visible_time[neighbor] <= timeid and deg[neighbor] > 0:
                        for second_neighbor in adj[neighbor, : ]:
                            if self.visible_time[second_neighbor] <= timeid and deg[second_neighbor] > 0:
                                for third_neighbor in adj[second_neighbor, : ]:
                                    if self.visible_time[third_neighbor] <= timeid:
                                        flag=1
                                        break
                            if flag==1:
                                break
                    if flag==1:
                        break
            if flag==1:
                reserved_session_ids.append(sessid)
        return reserved_session_ids,data.SessionId.nunique(),len(data)
    
    def _padding_sessions(self, data):
        '''
        Pad zeros at the end of each session to length self.max_length for batch training.
        '''
        ## Sort by the number of casualties
        data = data.sort_values(by=['TimeId','Rating','Timestamp'],ascending=[True,False,True]).groupby('SessionId')['ItemId'].apply(list).to_dict()
        ## Sort by time
        # data = data.sort_values(by=['TimeId','Timestamp','Rating'],ascending=[True,True,False]).groupby('SessionId')['ItemId'].apply(list).to_dict()
        new_data = {}
        data_mask = {}
        for k, v in data.items():
            mask = np.ones(self.max_length, dtype=np.float32)
            x = v[:-1]
            y = v[1: ]
            assert len(x) > 0
            padded_len = self.max_length - len(x)
            if padded_len > 0:
                x.extend([0] * padded_len)
                y.extend([0] * padded_len)
                mask[-padded_len: ] = 0.
            v.extend([0] * (self.max_length - len(v)))
            x = x[:self.max_length]
            y = y[:self.max_length]
            v = v[:self.max_length]
            new_data[k] = [np.array(x, dtype=np.int32), np.array(y, dtype=np.int32), np.array(v, dtype=np.int32)]
            data_mask[k] = np.array(mask, dtype=bool)
        return new_data, data_mask
    
    
    def _batch_feed_dict(self, current_batch):
        '''
        Construct batch inputs.
        '''
        current_batch_sess_ids, samples, support_sizes = current_batch
        self.current_batch_sess_ids=current_batch_sess_ids
        
        input_x = []
        input_y = []
        mask_y = []
        timeids = []
        grid_info_list=[]
        # flag=0
        for sessid in current_batch_sess_ids:
            nodeid, timeid = sessid.split('_')
            timeids.append(int(timeid))
            x, y, _ = self.padded_data[sessid]
            mask = self.mask[sessid]
            input_x.append(x)
            input_y.append(y)
            mask_y.append(mask)
            grid_info_list.append(self.dicts[timeid])
        support_layers_session = []
        support_layers_length = []
        for layer in range(self.num_layers):
            start = 0
            t = self.num_layers - layer
            support_sessions = []
            support_lengths = []
            for batch in range(self.batch_size):
                timeid = timeids[batch]
                support_nodes = samples[t][start: start + support_sizes[t]]
                for support_node in support_nodes:
                    support_session_id = str(self.latest_sessions[support_node][timeid])
                    support_session = self.padded_data[support_session_id][2]
                    #print(support_session)
                    length = np.count_nonzero(support_session)
                    support_sessions.append(support_session)
                    support_lengths.append(length)
                start += support_sizes[t]
            support_layers_session.append(support_sessions)
            support_layers_length.append(support_lengths)

        feed_dict = {}
        feed_dict.update({self.placeholders['input_x']: input_x})
        feed_dict.update({self.placeholders['input_y']: input_y})
        feed_dict.update({self.placeholders['mask_y']: mask_y})
        for i in range(self.num_layers):
            feed_dict.update({self.placeholders['support_sessions_layer%s'%(i+1)]:support_layers_session[i]})
            feed_dict.update({self.placeholders['support_lengths_layer%s'%(i+1)]:support_layers_length[i]})
        if self.num_layers==1:
            feed_dict.update({self.placeholders['support_nodes_layer1']: samples[1]})
        if self.num_layers==2:
            feed_dict.update({self.placeholders['support_nodes_layer1']: samples[2]})
            feed_dict.update({self.placeholders['support_nodes_layer2']: samples[1]})
        if self.num_layers==3:
            feed_dict.update({self.placeholders['support_nodes_layer1']: samples[3]})
            feed_dict.update({self.placeholders['support_nodes_layer2']: samples[2]})
            feed_dict.update({self.placeholders['support_nodes_layer3']: samples[1]})
                

        grid_info=np.dstack(grid_info_list)
        grid_info=grid_info.reshape(grid_info.shape[2],grid_info.shape[0],grid_info.shape[1])
        feed_dict.update({self.placeholders['grid_info']: grid_info})
        return feed_dict 

    def sample(self, nodeids, timeids, sampler):
        '''
        Sample neighbors recursively. First-order, then second-order, ...
        '''
        samples = [nodeids]
        support_size = 1
        support_sizes = [support_size]
        for k in range(self.num_layers):
            t = self.num_layers - k - 1
            node = sampler([samples[k], self.samples[t], timeids, support_size,self.num_layers,k])
            support_size *= self.samples[t]
            samples.append(np.reshape(node, [support_size * self.batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes

    def next_val_minibatch_feed_dict(self, valid_or_test='val'):
        '''
        Construct evaluation or test inputs.
        '''
        if valid_or_test == 'val':
            start = self.batch_num_val * self.batch_size
            self.batch_num_val += 1
            data = self.valid_session_ids
        elif valid_or_test == 'test':
            start = self.batch_num_test * self.batch_size
            self.batch_num_test += 1
            data = self.test_session_ids
        elif valid_or_test == 'predict':
            start = self.batch_num_predict * self.batch_size
            self.batch_num_predict += 1
            data = self.predict_session_ids
        else:
            raise NotImplementedError
        
        current_batch_sessions =data[start: start + self.batch_size]
        if len(current_batch_sessions)<self.batch_size:
            supplement_batch_sessions = np.random.choice(data, self.batch_size-len(current_batch_sessions), replace=True)
            current_batch_sessions=np.hstack([current_batch_sessions,supplement_batch_sessions])
        self.current_batch_sessions=current_batch_sessions
        nodes = [int(sessionid.split('_')[0]) for sessionid in current_batch_sessions]
        timeids =[int(sessionid.split('_')[1]) for sessionid in current_batch_sessions]
        samples, support_sizes = self.sample(nodes, timeids, self.test_sampler)
        return self._batch_feed_dict([current_batch_sessions, samples, support_sizes])
    
    
    def next_train_minibatch_feed_dict(self):
        '''
        Generate next training batch data.
        '''
        start = self.batch_num * self.batch_size
        self.batch_num += 1
        current_batch_sessions = self.train_session_ids[start: start + self.batch_size]
        if len(current_batch_sessions)<self.batch_size:
            supplement_batch_sessions = np.random.choice(self.train_session_ids, self.batch_size-len(current_batch_sessions), replace=True)
            current_batch_sessions=np.hstack([current_batch_sessions,supplement_batch_sessions])
        self.current_batch_sessions=current_batch_sessions

        nodes = [int(sessionid.split('_')[0]) for sessionid in current_batch_sessions]
        timeids = [int(sessionid.split('_')[1]) for sessionid in current_batch_sessions]
        samples, support_sizes = self.sample(nodes, timeids, self.sampler)
        return self._batch_feed_dict([current_batch_sessions, samples, support_sizes])

        
    def construct_adj(self):
        '''
        Construct adj table used during training.
        '''
        adj = self.num_nodes*np.ones((self.num_nodes+1, self.max_degree), dtype=np.int32)
        deg = np.zeros((self.num_nodes,))
        missed = 0
        for nodeid in self.train_df.UserId.unique():
            neighbors = np.array([neighbor for neighbor in 
                                self.adj_info.loc[self.adj_info['Follower']==nodeid].Followee.unique()], dtype=np.int32)
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                missed += 1
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        #print('Unexpected missing during constructing adj list: {}'.format(missed))
        return adj, deg

    def construct_test_adj(self):
        '''
        Construct adj table used during evaluation or testing.
        '''
        adj = self.num_nodes*np.ones((self.num_nodes+1, self.max_degree), dtype=np.int32)
        deg = np.zeros((self.num_nodes,))
        missed = 0
        data = self.all_data
        for nodeid in data.UserId.unique():
            neighbors = np.array([neighbor for neighbor in 
                                self.adj_info.loc[self.adj_info['Follower']==nodeid].Followee.unique()], dtype=np.int32)
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                missed += 1
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        #print('Unexpected missing during constructing adj list: {}'.format(missed))
        return adj, deg

    def end(self):
        '''
        Indicate whether we finish a pass over all training samples.
        '''
        return self.batch_num * self.batch_size > len(self.train_session_ids)# - self.batch_size
    
    def end_val(self, valid_or_test='val'):
        '''
        Indicate whether we finish a pass over all testing or evaluation samples.
        '''
        # batch_num = self.batch_num_val if valid_or_test == 'val' else self.batch_num_test
        # data = self.valid_session_ids if valid_or_test == 'val' else self.test_session_ids
        if valid_or_test == 'val':
            batch_num=self.batch_num_val
            data = self.valid_session_ids
        elif valid_or_test == 'test':
            batch_num=self.batch_num_test
            data = self.test_session_ids
        elif valid_or_test == 'predict':
            batch_num=self.batch_num_predict
            data = self.predict_session_ids

        end = batch_num * self.batch_size > len(data)# - self.batch_size
        if end:
            if valid_or_test == 'val':
                self.batch_num_val = 0
            elif valid_or_test == 'test':
                self.batch_num_test = 0
            elif valid_or_test == 'predict':
                self.batch_num_predict = 0
            else:
                raise NotImplementedError
        if end:
            self.batch_num_val = 0
        return end

    def shuffle(self):
        '''
        Shuffle training data.
        '''
        self.train_session_ids = np.random.permutation(self.train_session_ids)
        self.batch_num = 0
    
