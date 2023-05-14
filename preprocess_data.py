#coding=utf-8
import os
import random
import itertools
import pandas as pd
import numpy as np

import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
tf.disable_eager_execution()
from functools import reduce


class Args:
    def __init__(self,max_length,friends_long_short_term,grid_feature,friend_layer_num,type_flag):
        self.training = True
        self.epochs = 100
        self.aggregator_type='attn'
        self.act='relu'
        self.batch_size = 200
        self.max_degree = 50 # 最大连接的关系
        self.num_users = -1
        self.num_items = 100
        self.concat=False
        self.learning_rate=0.001
        self.hidden_size = 100
        self.embedding_size = 100
        self.emb_user = 100
        self.max_length=max_length
        self.model_size = 'small'
        self.dropout = 0.2
        self.weight_decay = 0.
        self.print_every = 100
        self.val_every = 500
        self.ckpt_dir = './save/'
        self.decay_steps=400
        self.decay_rate=0.98
        self.num_feature=-1
        self.type_flag=type_flag
        if friends_long_short_term=='1&0':
            self.global_only = True
            self.local_only = False
        elif friends_long_short_term=='0&1':
            self.global_only = False
            self.local_only = True
        elif friends_long_short_term=='1&1':
            self.global_only = False
            self.local_only = False
        self.grid_feature=grid_feature
        if type_flag in ['baseline','self']:
            self.friend_layer_num=1
            self.friends_long_short_term='1&1'
        else:
            self.friend_layer_num=friend_layer_num
        if self.friend_layer_num==1:
            self.samples=[10]
            self.dims=[200]
        elif self.friend_layer_num==2:
            self.samples=[5,10]
            self.dims=[100,100]
        elif self.friend_layer_num==3:
            self.samples=[2,5,10]
            self.dims=[50,50,100]
            
def construct_placeholders(args):
    # Define placeholders
    placeholders = {
            'input_x': tf.placeholder(tf.int32, shape=(args.batch_size, args.max_length), name='input_session'),
            'input_y': tf.placeholder(tf.int32, shape=(args.batch_size, args.max_length), name='output_session'),
            'mask_y': tf.placeholder(tf.float32, shape=(args.batch_size, args.max_length), name='mask_x'),
        }
    n=reduce(lambda x,y:x*y,args.samples)
    placeholders['support_nodes_layer1']=tf.placeholder(tf.int32,shape=(args.batch_size*n),name='support_nodes_layer1')
    placeholders['support_sessions_layer1']=tf.placeholder(tf.int32,shape=(args.batch_size*n,args.max_length),name='support_sessions_layer1')
    placeholders['support_lengths_layer1']=tf.placeholder(tf.int32, shape=(args.batch_size*n),name='support_lengths_layer1')          
    if args.friend_layer_num==2:
        m=args.samples[0]
        placeholders['support_nodes_layer2']=tf.placeholder(tf.int32, shape=(args.batch_size*m),name='support_nodes_layer2')
        placeholders['support_sessions_layer2']=tf.placeholder(tf.int32, shape=(args.batch_size*m,args.max_length),name='support_sessions_layer2')
        placeholders['support_lengths_layer2']=tf.placeholder(tf.int32, shape=(args.batch_size*m),name='support_lengths_layer2')
    elif args.friend_layer_num==3:
        m=args.samples[0]*args.samples[1]
        placeholders['support_nodes_layer2']=tf.placeholder(tf.int32, shape=(args.batch_size*m),name='support_nodes_layer2')
        placeholders['support_sessions_layer2']=tf.placeholder(tf.int32, shape=(args.batch_size*m,args.max_length),name='support_sessions_layer2')
        placeholders['support_lengths_layer2']=tf.placeholder(tf.int32, shape=(args.batch_size*m),name='support_lengths_layer2')
        p=args.samples[0]
        placeholders['support_nodes_layer3']=tf.placeholder(tf.int32, shape=(args.batch_size*p),name='support_nodes_layer3')
        placeholders['support_sessions_layer3']=tf.placeholder(tf.int32, shape=(args.batch_size*p,args.max_length),name='support_sessions_layer3')
        placeholders['support_lengths_layer3']=tf.placeholder(tf.int32, shape=(args.batch_size*p),name='support_lengths_layer3')
    if args.grid_feature:
        placeholders['grid_info']=tf.placeholder(tf.float32, shape=(args.batch_size,args.num_items-1,args.num_feature), name='grid_info')
    return placeholders

def get_df_attack(social_net,interval,period,item,floder):
    file=floder+'%s/%s_degree/%s/attacks_%s.pkl'%(social_net,interval,period,item)
    if os.path.exists(file):
        df_attack=pd.read_pickle(file)
        ## check
        day_=pd.Timestamp(2019,10,31)
        timestamp_=pd.Timestamp.timestamp(day_)
        index=df_attack['Timestamp']<=timestamp_
        df_attack=df_attack[index].reset_index(drop=True)
    else:
        df=pd.read_csv('/home/wujj/Code/Terrorist_Pattern/Data_/04_实验数据/01_恐袭数据/GTD_Pre_Data.csv')
        if item=='grid':
            df['ItemId']=df['%s_grid'%interval]
        else:
            item_=item.split('_')[1]
            index=df[item_].isnull()
            df.loc[index,item_]=0
            df[item_]=df[item_].astype(int)
            n=len(str(df[item_].max()))
            df['ItemId']=df['%s_grid'%interval]*(10**n)+df[item_]
            index=(df['guncertain1']!=1)&(df['guncertain2']!=1)&(df['guncertain3']!=1)
            df=df[index].reset_index(drop=True)
        df['casualties']=df['nwound']+df['nkill']
        index=df['casualties'].isnull()
        df.loc[index,'casualties']=0
        df_attack=pd.DataFrame(columns=['UserId','ItemId','TimeId','Timestamp','Rating','Grid'])
        for i in range(len(df)):
            ItemId=df.loc[i,'ItemId']
            TimeId=df.loc[i,'%s_TimeId'%period]
            Timestamp=df.loc[i,'timestamp']
            Rating=df.loc[i,'casualties']
            Grid=df.loc[i,'%s_grid'%interval]
            
            for j in ['gname','gname2','gname3']:
                UserId=df.loc[i,j]
                if not np.isnan(UserId):
                    df_attack.loc[len(df_attack)]=[UserId,ItemId,TimeId,Timestamp,Rating,Grid]
        df_attack_final=pd.DataFrame(columns=['UserId','ItemId','TimeId','Timestamp','Rating','Grid'])
        for UserId in df_attack['UserId'].unique():
            index1=df_attack['UserId']==UserId
            for TimeId in df_attack[index1]['TimeId'].unique():
                index2=index1&(df_attack['TimeId']==TimeId)
                for ItemId in df_attack[index2]['ItemId'].unique():
                    index3=index2&(df_attack['ItemId']==ItemId)
                    for Timestamp in df_attack[index3]['Timestamp'].unique():
                        index4=index3&(df_attack['Timestamp']==Timestamp)
                        if len(df_attack[index4])>0:
                            if len(df_attack[index4])==1:
                                df_attack_final=df_attack_final.append(df_attack[index4])
                            else:
                                df_attack_final.loc[len(df_attack_final)]=[UserId,ItemId,TimeId,df_attack[index4]['Timestamp'].min(),df_attack[index4]['Rating'].sum(),df_attack[index4]['Grid'].mean()]
                    # if len(df_attack[index3])>0:
                    #     if len(df_attack[index3])==1:
                    #         df_attack_final=df_attack_final.append(df_attack[index3])
                    #     else:
                    #         df_attack_final.loc[len(df_attack_final)]=[UserId,ItemId,TimeId,df_attack[index3]['Timestamp'].min(),df_attack[index3]['Rating'].sum(),df_attack[index3]['Grid'].mean()]
        df_attack=df_attack_final
        df_attack['UserId']=df_attack['UserId'].astype('int').astype('str')
        df_attack['ItemId']=df_attack['ItemId'].astype('int').astype('str')
        df_attack['TimeId']=df_attack['TimeId'].astype('int32')
        df_attack['Rating']=df_attack['Rating'].astype('int32')
        df_attack['Timestamp']=df_attack['Timestamp'].astype('float32')
        df_attack['Grid']=df_attack['Grid'].astype('int').astype('str')
        df_attack.reset_index(drop=True).to_pickle(file)
    return df_attack


def get_df_net(social_net,net_shuffle,df_attack):
    with open('/data/Experiment/random_seed.txt','r') as f:
        seed=int(f.read())
    random.seed(seed)
    file='/home/wujj/Code/Terrorist_Pattern/Data_/04_实验数据/02_关系网/socialnet_%s.tsv'%social_net
    df_net=pd.read_csv(file, sep='\t',dtype={0:str, 1: str})
    if net_shuffle:
        index=df_net['Follower'].isin(df_attack['UserId'].unique())
        comb_group=list(itertools.combinations(df_net[index]['Follower'].unique(),2))
        df_adj=pd.DataFrame(columns=['Follower','Followee','Weight'])
        for comb in random.sample(comb_group, int(len(df_net)/2)):
            id1=comb[0]
            id2=comb[1]
            if id1!=id2:
                df_adj.loc[df_adj.shape[0]]=[id1,id2,1]
                df_adj.loc[df_adj.shape[0]]=[id2,id1,1]
        df_net=df_adj
        df_net.to_csv('/home/wujj/Code/Terrorist_Pattern/Data_/04_实验数据/02_关系网/socialnet_%s_shuffle.tsv'%social_net,sep='\t',index=0)
    friend_size=df_net.groupby('Follower').size()
    index=df_net['Follower'].isin(friend_size[friend_size>=1].index) # 至少有n个关系
    df_net=df_net[index].reset_index(drop=True)
    return df_net


def reset_id(data, id_map, column_name='UserId'):
    mapped_id=data[column_name].map(id_map)
    data[column_name]=mapped_id
    if column_name == 'UserId':
        session_id=[str(uid)+'_'+str(tid) for uid, tid in zip(data['UserId'], data['TimeId'])]
        data['SessionId']=session_id
    return data

def split_data(parameter):
    with open('/data/Experiment/random_seed.txt','r') as f:
        seed=int(f.read())
    np.random.seed(seed)
    try:
        floder='/data/Experiment/'
        social_net,interval,period,max_length,grid_feature,friends_long_short_term,item,net_shuffle,friend_layer_num,type_flag=parameter
        result_floder=floder+'%s/%s_degree/%s/%s_net_shuffle_%s/'%(social_net,interval,period,item,net_shuffle)
        df_attack=get_df_attack(social_net,interval,period,item,floder)
        session_id=[str(uid)+'_'+str(tid) for uid, tid in zip(df_attack['UserId'].astype(int), df_attack['TimeId'].astype(int))]
        df_attack['SessionId']=session_id
        df_net=get_df_net(social_net,net_shuffle,df_attack)
        # 关系网中Follower和Followee都需要在行动记录中出现
        df_net=df_net.loc[df_net['Follower'].isin(df_attack['UserId'].unique())]
        df_net=df_net.loc[df_net['Followee'].isin(df_attack['UserId'].unique())]
        # 两者求交集
        df_attack=df_attack.loc[df_attack['UserId'].isin(df_net.Follower.unique())].reset_index(drop=True)
        # SessionId至少出现两次,两次以上才能形成序列
        df_attack=df_attack[df_attack.groupby('SessionId')['SessionId'].transform('size')>1].reset_index(drop=True)
        tmax=df_attack.TimeId.max()
        tmin=df_attack.TimeId.min()
        session_max_times=df_attack.groupby('SessionId').TimeId.max()
        # 排除只有一个SessionId且位于最后一个TimeId的UserId
        user2sessions=df_attack.groupby('UserId')['SessionId'].apply(set).apply(len)
        index1=df_attack['UserId'].isin(user2sessions[user2sessions<2].index)
        index1=df_attack['UserId'].isin(user2sessions[user2sessions<2].index)
        index2=df_attack[index1]['TimeId']==tmax
        if len(df_attack[index1][index2])>0:
            index3=index1&index2
            df_attack=df_attack[~index3]
        # 训练集占80%
        num=int((tmax-tmin+1)*0.2)
        session_train=session_max_times[session_max_times < tmax - num].index
        session_holdout=session_max_times[session_max_times >= tmax - num].index
        train_set=df_attack[df_attack['SessionId'].isin(session_train)] 
        holdout_set=df_attack[df_attack['SessionId'].isin(session_holdout)]
        # 在训练集中ItemId至少出现两次
        train_set=train_set[train_set.groupby('ItemId')['ItemId'].transform('size')>1] 
        # 在训练集中SessionId至少出现两次,两次以上才能形成序列
        train_set=train_set[train_set.groupby('SessionId')['SessionId'].transform('size')>1]
        # 在训练集中某个UserId的SessionId至少出现两次,两次以上才能挖掘行动规律（长期和短期）
        train_set=train_set[train_set.groupby('UserId')['SessionId'].transform('nunique')>1]
        item_to_predict=train_set['ItemId'].unique()
        holdout_cn=holdout_set.SessionId.nunique()
        holdout_ids=holdout_set.SessionId.unique()
        np.random.shuffle(holdout_ids)
        # 验证集和测试集各占剩下20%的一半
        valid_cn=int(holdout_cn * 0.5)
        session_valid=holdout_ids[0: valid_cn]
        session_test=holdout_ids[valid_cn: ]
        valid_set=holdout_set[holdout_set['SessionId'].isin(session_valid)]
        test_set=holdout_set[holdout_set['SessionId'].isin(session_test)]
        # 在验证集和测试集中SessionId只有一个的话不能出现在最后一个SessionId
        valid_set=valid_set[valid_set['ItemId'].isin(item_to_predict)]
        valid_set=valid_set[valid_set.groupby('SessionId')['SessionId'].transform('size')>1]
        valid_set=valid_set[valid_set.groupby('UserId')['TimeId'].transform('min')!=tmax]
        test_set=test_set[test_set['ItemId'].isin(item_to_predict)]
        test_set=test_set[test_set.groupby('SessionId')['SessionId'].transform('size')>1]
        test_set=test_set[test_set.groupby('UserId')['TimeId'].transform('min')!=tmax]
        total_df=pd.concat([train_set, valid_set, test_set])
        df_net=df_net.loc[df_net['Follower'].isin(total_df['UserId'].unique())]
        df_net=df_net.loc[df_net['Followee'].isin(total_df['UserId'].unique())]
        user_map=dict(zip(total_df.UserId.unique(), range(total_df.UserId.nunique()))) 
        item_map=dict(zip(total_df.ItemId.unique(), range(1, 1+total_df.ItemId.nunique())))
        with open(result_floder+'user_id_map.tsv', 'w') as f:
            for k, v in user_map.items():
                t=f.write(str(k) + '\t' + str(v) + '\n')
        with open(result_floder+'item_id_map.tsv', 'w') as f:
            for k, v in item_map.items():
                t=f.write(str(k) + '\t' + str(v) + '\n')
        num_users=len(user_map)
        num_items=len(item_map)
        total_df=reset_id(total_df, user_map)
        train_set=reset_id(train_set, user_map)
        valid_set=reset_id(valid_set, user_map)
        test_set=reset_id(test_set, user_map)
        df_net=reset_id(df_net, user_map, 'Follower')
        df_net=reset_id(df_net, user_map, 'Followee')
        total_df=reset_id(total_df, item_map, 'ItemId')
        train_set=reset_id(train_set, item_map, 'ItemId')
        valid_set=reset_id(valid_set, item_map, 'ItemId')
        test_set=reset_id(test_set, item_map, 'ItemId')
        user2sessions=total_df.groupby('UserId')['SessionId'].apply(set).to_dict()
        user_latest_session=[]
        for idx in range(num_users):
            sessions=user2sessions[idx]
            latest=[]
            for t in range(tmax+1):
                if t == 0:
                    latest.append('NULL')
                else:
                    sess_id_tmp=str(idx) + '_' + str(t-1)
                    if sess_id_tmp in sessions:
                        latest.append(sess_id_tmp)
                    else:
                        latest.append(latest[t-1])
            if np.unique(np.array(latest)).shape[0]<=1:
                print(tmax,parameter,np.unique(np.array(latest)),user2sessions[idx],sess_id_tmp)
            user_latest_session.append(latest)
        train_set.to_csv(result_floder+'train.tsv', sep='\t', index=0)
        valid_set.to_csv(result_floder+'valid.tsv', sep='\t', index=0)
        test_set.to_csv(result_floder+'test.tsv', sep='\t', index=0)
        df_net.to_csv(result_floder+'adj.tsv', sep='\t', index=0)
        with open(result_floder+'latest_sessions.txt', 'w') as f:
            for idx in range(num_users):
                t=f.write(','.join(user_latest_session[idx]) + '\n')
    except Exception as ex:
        print(ex.__traceback__.tb_lineno,ex,parameter)

# parameter=('GTD',0.1,'1week',20,1,'11','grid',0)
# df=split_data(parameter)