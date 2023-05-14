import time
from preprocess_data import *
from itertools import product
from utils import load_data
from minibatch import *
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
tf.disable_eager_execution()
from model import DGRec
tf.test.is_gpu_available()
tf.config.list_physical_devices("GPU")
import random
seed=3
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(3)
k=10
social_net='EDTG'#['GTD','BAAD','EDTG','Pattern']
interval=0.5
period='1week'#['1week','2weeks','1month','2months']
max_length=20
grid_feature=0
friends_long_short_term='1&1'
item=['grid','grid_attacktype1','grid_targtype1','grid_natlty1','grid_weaptype1']#[]
net_shuffle=[0]
friend_layer_num=[2]
type_flag=['self','social','self&social']
choices=list(product(social_net,interval,period,max_length,grid_feature,friends_long_short_term,item,net_shuffle))
n=len(choices)
n
parameter=('EDTG', 0.1, '2months', 20, 1, '1&0', 'grid', 0, 1, 'self&social')
parameter
social_net,interval,period,max_length,grid_feature,friends_long_short_term,item,net_shuffle,friend_layer_num,type_flag=parameter