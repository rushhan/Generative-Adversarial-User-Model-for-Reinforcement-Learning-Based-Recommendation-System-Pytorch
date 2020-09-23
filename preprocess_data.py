
import pandas as pd

import numpy as np
import pickle
import pandas as pd
import argparse
from past.builtins import xrange

from options import get_options


def main(opts):

	filename = opts.data_folder+opts.dataset+'.txt'

	raw_data = pd.read_csv(filename, sep='\t', usecols=[1, 3, 5, 7, 6], dtype={1: int, 3: int, 7: int, 5:int, 6:int})


	raw_data.drop_duplicates(subset=['session_new_index','Time','item_new_index','is_click'], inplace=True)
	
	raw_data.sort_values(by='is_click',inplace=True)
	print (raw_data.head())
	raw_data.drop_duplicates(keep='last', subset=['session_new_index','Time','item_new_index'], inplace=True)

	sizes = raw_data.nunique()
	print (sizes)
	size_user = sizes['session_new_index']
	size_item = sizes['item_new_index']

	data_user = raw_data.groupby(by='session_new_index')
	print (data_user)
	data_behavior = [[] for _ in xrange(size_user)]

	train_user = []
	vali_user = []
	test_user = []

	sum_length = 0
	event_cnt = 0

	for user in xrange(size_user):
	    data_behavior[user] = [[], [], []]
	    data_behavior[user][0] = user
	    data_u = data_user.get_group(user)
	    split_tag = list(data_u['tr_val_tst'])[0]
	    if split_tag == 0:
	        train_user.append(user)
	    elif split_tag == 1:
	        vali_user.append(user)
	    else:
	        test_user.append(user)

	    data_u_time = data_u.groupby(by='Time')
	    time_set = np.array(list(set(data_u['Time'])))
	    time_set.sort()

	    true_t = 0
	    for t in xrange(len(time_set)):
	        display_set = data_u_time.get_group(time_set[t])
	        event_cnt += 1
	        sum_length += len(display_set)

	        data_behavior[user][1].append(list(display_set['item_new_index']))
	        data_behavior[user][2].append(int(display_set[display_set.is_click==1]['item_new_index']))

	new_features = np.eye(size_item)

	filename = opts.data_folder+opts.dataset+'.pkl'
	file = open(filename, 'wb')
	print (data_behavior)
	pickle.dump(data_behavior, file, protocol=pickle.HIGHEST_PROTOCOL)
	pickle.dump(new_features, file, protocol=pickle.HIGHEST_PROTOCOL)
	file.close()

	filename = opts.data_folder+opts.dataset+'-split.pkl'
	file = open(filename, 'wb')
	pickle.dump(train_user, file, protocol=pickle.HIGHEST_PROTOCOL)
	pickle.dump(vali_user, file, protocol=pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_user, file, protocol=pickle.HIGHEST_PROTOCOL)
	file.close()




if __name__ == "__main__":
    main(get_options())