# define the main training and inference here
import sys
sys.path
sys.path.append('../../')

import torch
from torch import nn
#import Dataloader
import tqdm
from options import get_options
from torch.utils.data import DataLoader
import tqdm
import os
import json
import pprint as pp
from model import UserModelPW
from data_utils import *
import torch.optim as optim


import datetime
import numpy as np
import os
import threading

#train,test,val=get_split(ops.Datadir)

#train_input = get_data(train)
#val_input = get_data(val)
#test_input = get_data(test)



'''
@for i in range(pot.iter):
    train_model


    if iter//some_fixed_no==0:
        validate

test_model 
plot_figs
'''

def multithread_compute_vali(opts,valid_data,model):
    global vali_sum, vali_cnt

    vali_sum = [0.0, 0.0, 0.0]
    vali_cnt = 0
    threads = []
    for ii in xrange(opts.num_thread):
        #print ("got here")
        #print (dataset.model_type)
        #print (" [dataset.vali_user[ii]]", [dataset.vali_user[ii]])
        #valid_data = dataset.prepare_validation_data(1, [dataset.vali_user[15]]) # is a dict

       # print ("valid_data",valid_data)
        #sys.exit()

        thread = threading.Thread(target=vali_eval, args=(1, ii,opts,valid_data,model))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    

    return vali_sum[0]/vali_cnt, vali_sum[1]/vali_cnt, vali_sum[2]/vali_cnt

lock = threading.Lock()


def vali_eval(xx, ii,opts,valid_data,model):
    global vali_sum, vali_cnt
    #print ("dataset.vali_user",dataset.vali_user)
    
    #valid_data = dataset.prepare_validation_data(1, [dataset.vali_user[ii]]) # is a dict

    #print ("valid_data",valid_data)
    #sys.exit()
    with torch.no_grad():
        _,_,_, loss_sum, precision_1_sum, precision_2_sum, event_cnt = model(valid_data,index=ii)

    lock.acquire()
    vali_sum[0] += loss_sum
    vali_sum[1] += precision_1_sum
    vali_sum[2] +=precision_2_sum
    vali_cnt += event_cnt
    lock.release()


lock = threading.Lock()



def multithread_compute_test(opts,test_data,model):
    global test_sum, test_cnt

    num_sets = 1 * opts.num_thread

    thread_dist = [[] for _ in xrange(opts.num_thread)]
    for ii in xrange(num_sets):
        thread_dist[ii % opts.num_thread].append(ii)

    test_sum = [0.0, 0.0, 0.0]
    test_cnt = 0
    threads = []
    for ii in xrange(opts.num_thread):
        thread = threading.Thread(target=test_eval, args=(1, thread_dist[ii],opts,test_data,model))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return test_sum[0]/test_cnt, test_sum[1]/test_cnt, test_sum[2]/test_cnt


def test_eval(xx, thread_dist,opts,test_data,model):
    global test_sum, test_cnt
    test_thread_eval = [0.0, 0.0, 0.0]
    test_thread_cnt = 0
    for ii in thread_dist:
        
        with torch.no_grad():
            _,_,_, loss_sum, precision_1_sum, precision_2_sum, event_cnt = model(test_data,index=ii)

        test_thread_eval[0] += loss_sum
        test_thread_eval[1] +=precision_1_sum
        test_thread_eval[2] += precision_2_sum
        test_thread_cnt += event_cnt

    lock.acquire()
    test_sum[0] += test_thread_eval[0]
    test_sum[1] += test_thread_eval[1]
    test_sum[2] += test_thread_eval[2]
    test_cnt += test_thread_cnt
    lock.release()



def init_weights(m):
    sd = 1e-3
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight)
        m.weight.data.clamp_(-sd,sd) # to mimic the normal clmaped weight initilization


def main(opts):
    pp.pprint(vars(opts))

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start" % log_time)

    dataset = Dataset(opts)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, load data completed" % log_time)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare vali data" % log_time)


    valid_data=dataset.prepare_validation_data(opts.num_thread, dataset.vali_user)
   
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare validation data, completed" % log_time)

    
    model = UserModelPW(dataset.f_dim, opts)
    model.apply(init_weights)
    
    #optimizer = optim.Adam(
    #   [{'params': model.parameters(), 'lr': opts.learning_rate}])

    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate, betas=(0.5, 0.999))


    best_metric = [100000.0, 0.0, 0.0]

    vali_path = opts.save_dir+'/'
    if not os.path.exists(vali_path):
        os.makedirs(vali_path)


    #training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1) # need to change the dataloader

    for i in xrange(opts.num_itrs):

        #model.train()
        for p in model.parameters():
        	p.requires_grad = True
        model.zero_grad()

        training_user_nos = np.random.choice(dataset.train_user, opts.batch_size, replace=False)

        training_user= dataset.data_process_for_placeholder(training_user_nos)
        for p in model.parameters():
            p.data.clamp_(-1e0, 1e0)

        #for batch_id, batch in enumerate(tqdm(training_dataloader)): # the original code does not iterate over entire batch , so change this one
        
        loss,_,_,_,_,_,_= model(training_user,is_train=True)
        #print ("the loss is",loss)
        
        loss.backward() 
        optimizer.step()
        
        if np.mod(i, 10) == 0:
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, start first iteration validation" % log_time)

        if np.mod(i, 10) == 0:
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, start first iteration validation" % log_time)
            vali_loss_prc = multithread_compute_vali(opts,valid_data,model)
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, first iteration validation complete" % log_time)

            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: itr%d, vali: %.5f, %.5f, %.5f" %
                  (log_time, i, vali_loss_prc[0], vali_loss_prc[1], vali_loss_prc[2]))

            if vali_loss_prc[0] < best_metric[0]:
                best_metric[0] = vali_loss_prc[0]
                best_save_path = os.path.join(vali_path, 'best-loss')
                torch.save(model.state_dict(), best_save_path)
                #best_save_path = saver.save(sess, best_save_path)
            if vali_loss_prc[1] > best_metric[1]:
                best_metric[1] = vali_loss_prc[1]
                best_save_path = os.path.join(vali_path, 'best-pre1')
                torch.save(model.state_dict(), best_save_path)
            if vali_loss_prc[2] > best_metric[2]:
                best_metric[2] = vali_loss_prc[2]
                best_save_path = os.path.join(vali_path, 'best-pre2')
                torch.save(model.state_dict(), best_save_path)

        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, iteration %d train complete" % (log_time, i))

    # test
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare test data" % log_time)

    test_data = dataset.prepare_validation_data(opts.num_thread, dataset.test_user)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare test data end" % log_time)


    best_save_path = os.path.join(vali_path, 'best-loss')
    model.load_state_dict(torch.load(best_save_path))
    #saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test(opts,test_data,model)
    vali_loss_prc = multithread_compute_vali(opts,valid_data,model)
    print("test!!!loss!!!, test: %.5f, vali: %.5f" % (test_loss_prc[0], vali_loss_prc[0]))

    best_save_path = os.path.join(vali_path, 'best-pre1')
    model.load_state_dict(torch.load(best_save_path))
    #saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test(opts,test_data,model)
    vali_loss_prc = multithread_compute_vali(opts,valid_data,model)
    print("test!!!pre1!!!, test: %.5f, vali: %.5f" % (test_loss_prc[1], vali_loss_prc[1]))

    best_save_path = os.path.join(vali_path, 'best-pre2')
    model.load_state_dict(torch.load(best_save_path))
    #saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test(opts,test_data,model)
    vali_loss_prc = multithread_compute_vali(opts,valid_data,model)
    print("test!!!pre2!!!, test: %.5f, vali: %.5f" % (test_loss_prc[2], vali_loss_prc[2]))




















if __name__ == "__main__":
    main(get_options())