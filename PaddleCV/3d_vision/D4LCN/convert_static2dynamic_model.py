#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""main """

import os
import sys
import time
import shutil
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from models import *
from utils import *
#from data.data_utils import *

sys.path.append(os.getcwd())
import lib.core as core
from lib.rpn_util import *
from data.m3drpn_reader import M3drpnReader#M3dRpn_Reader
import pdb


import paddle
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid import framework
import math
import time


logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    """parse"""
    parser = argparse.ArgumentParser("M3D-RPN train script")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use data parallel mode to train the model."
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='DenseNet121',
        help='backbone model to train, default MSG')
    parser.add_argument(
        '--conf',
        type=str,
        default='kitti_3d_multi_main',
        help='config')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='training batch size, default 16')
    
    # parser.add_argument(
    #     '--num_classes',
    #     type=int,
    #     default=2,
    #     help='number of classes in dataset, default: 40')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='initial learning rate, default 0.01')
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.7,
        help='learning rate decay gamma, default 0.5')
    parser.add_argument(
        '--bn_momentum',
        type=float,
        default=0.99,
        help='initial batch norm momentum, default 0.99')
    parser.add_argument(
        '--decay_steps',
        type=int,
        default=12500,
        help='learning rate and batch norm momentum decay steps, default 12500')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='L2 regularization weight decay coeff, default 1e-5.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=201,
        help='epoch number. default 201.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset/ModelNet40/modelnet40_ply_hdf5_2048',
        help='dataset directory')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_cls',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval for logging.')
    parser.add_argument(
        '--ce',
        action='store_true',
        help='The flag indicating whether to run the task '
        'for continuous evaluation.')
    parser.add_argument(
        '--pretrain',
        #action='store_true',
        default="pretrained_model/densenet121", 
        help='static backbone model path')
    
    args = parser.parse_args()
    return args

def save_vars(executor, dirname, program=None, vars=None):
    """
    Temporary resolution for Win save variables compatability.

    """

    save_program = fluid.Program()
    save_block = save_program.global_block()

    for each_var in vars:
        # NOTE: don't save the variable which type is RAW
        if each_var.type == fluid.core.VarDesc.VarType.RAW:
            continue
        new_var = save_block.create_var(
            name=each_var.name,
            shape=each_var.shape,
            dtype=each_var.dtype,
            type=each_var.type,
            lod_level=each_var.lod_level,
            persistable=True)
        file_path = os.path.join(dirname, new_var.name)
        file_path = os.path.normpath(file_path)
        save_block.append_op(
            type='save',
            inputs={'X': [new_var]},
            outputs={},
            attrs={'file_path': file_path})

    executor.run(save_program)


def save_checkpoint(exe, program, path):
    """
    Save checkpoint for evaluation or resume training
    """
    ckpt_dir = path
    print("Save model checkpoint to {}".format(ckpt_dir))
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    save_vars(
        exe,
        ckpt_dir,
        program,
        vars=list(filter(fluid.io.is_persistable, program.list_vars())))

    return ckpt_dir


def train():
    """main train"""
    args = parse_args()
    
    #print_arguments(args)
    # check whether the installed paddle is compiled with GPU
    #check_gpu(args.use_gpu)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    assert args.backbone in ['DenseNet121', 'ResNet101'], \
            "--backbone unsupported" 

    # pdb.set_trace()
    conf = core.init_config(args.conf)
    paths = core.init_training_paths(args.conf)

    # get reader and anchor
    m3drpn_reader = M3drpnReader(conf, args.data_dir)
    print("conf.batch_size",conf.batch_size)
    train_reader = m3drpn_reader.get_reader(conf.batch_size, mode='train')
    generate_anchors(conf, m3drpn_reader.data['train'], paths.output)

    # train
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)

    print("pretrain path:", args.pretrain)
    pretrain = fluid.load_program_state(args.pretrain)
    print("load ok")
    pdb.set_trace()
    pretrain_list = sorted(pretrain.items())#, key=operator.itemgetter(1)) 
    pretrain = {pretrain_data[0]: pretrain_data[1] for pretrain_data in pretrain_list}
    print("backbone-static keys: ", pretrain.keys())

    with fluid.dygraph.guard(place):
        if args.ce:
            print("ce mode")
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        
        train_model, optimizer = core.init_training_model(conf, args.backbone, paths.output)
        if args.use_data_parallel:
            train_model = fluid.dygraph.parallel.DataParallel(train_model, strategy)

        
        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)
        #TODO
        #load_vars, load_fail_vars = core.load_vars(train_prog, conf.pretrained)
        state_dict = train_model.state_dict()
        pdb.set_trace()
        print("backbone-dynamic keys: ", state_dict.keys())
        #TODO save pdparams
        #pretrain = fluid.load_program_state("pretrained_model/DenseNet121_pretrained")
        #state_dict = train_model.state_dict()
        #Convert
        #train_model.set_dict()

        total_batch_num = 0

        for eop in range(args.epoch):
            
            total_loss = 0.0
            total_acc1 = 0.0
            
            total_sample = 0
            
            for batch_id, data in enumerate(train_reader()):
                #print("data", data)
                #NOTE: used in benchmark
                # if args.max_iter and total_batch_num == args.max_iter:
                #     return
                batch_start = time.time()
                
                dy_x_data = np.array(
                    [x[0].reshape(3, 512, 1760) for x in data]).astype('float32')
                # if len(np.array([x[1]
                #                  for x in data]).astype('int64')) != batch_size:
                #     continue
                # y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                #     -1, 1)

                img = to_variable(dy_x_data)
                # label = to_variable(y_data)
                # label.stop_gradient = True

                out = train_model(img)
                
                # loss = fluid.layers.cross_entropy(input=out, label=label)
                # avg_loss = fluid.layers.mean(x=loss)

                # acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                

                # dy_out = avg_loss.numpy()

                # if args.use_data_parallel:
                #     avg_loss = train_model.scale_loss(avg_loss)
                #     avg_loss.backward()
                #     train_model.apply_collective_grads()
                # else:
                #     avg_loss.backward()

                # optimizer.minimize(avg_loss)
                # train_model.clear_gradients()

                batch_end = time.time()
                train_batch_cost = batch_end - batch_start
                # total_loss += dy_out
                # total_acc1 += acc_top1.numpy()
                
                total_sample += 1
                total_batch_num = total_batch_num + 1 #this is for benchmark
                #print("epoch id: %d, batch step: %d, loss: %f" % (eop, batch_id, dy_out))
                if batch_id % 10 == 0:
                    print( "epoch %d | batch step %d , batch cost: %.5f" % \
                           ( eop, batch_id, train_batch_cost))
                    # print( "epoch %d | batch step %d, loss %0.3f acc1 %0.3f , batch cost: %.5f" % \
                    #        ( eop, batch_id, total_loss / total_sample, \
                    #          total_acc1 / total_sample,  train_batch_cost))

            # if args.ce:
            #     print("kpis\ttrain_acc1\t%0.3f" % (total_acc1 / total_sample))
               
            #     print("kpis\ttrain_loss\t%0.3f" % (total_loss / total_sample))
            # print("epoch %d | batch step %d, loss %0.3f acc1 %0.3f" % \
            #       (eop, batch_id, total_loss / total_sample, \
            #        total_acc1 / total_sample))
            

            # save_parameters = (not args.use_data_parallel) or (
            #     args.use_data_parallel and
            #     fluid.dygraph.parallel.Env().local_rank == 0)
            # if save_parameters:
            #     fluid.save_dygraph(train_model.state_dict(),
            #                                     '_params')

if __name__ == '__main__':

    train()

    """
    
    if args.resume:
        if not os.path.isdir(args.resume):
            assert os.path.exists("{}.pdparams".format(args.resume)), \
                    "Given resume weight {}.pdparams not exist.".format(args.resume)
            assert os.path.exists("{}.pdopt".format(args.resume)), \
                    "Given resume optimizer state {}.pdopt not exist.".format(args.resume)
        fluid.load(train_prog, args.resume, exe)
        
    # pretrain 
    # if conf.pretrained:
    #     if os.path.exists(conf.pretrained):
    #         print('Pretrained model dir: ', conf.pretrained)
    #         load_vars, load_fail_vars = core.load_vars(train_prog, conf.pretrained)

    #         # loc
    #         program_state = fluid.load_program_state(conf.pretrained)
    #         loc_dict = {}
    #         for key in program_state.keys():
    #             if key.startswith("rpn_"):
    #                 loc_dict[key+"_loc"] = np.repeat(program_state[key], conf.bins, axis=0)
    #         fluid.set_program_state(train_prog, loc_dict)
    #         for key in loc_dict.keys():
    #             print("Parameter[{}] loaded sucessfully!".format(key))
            
    #         # matched   
    #         fluid.io.load_vars(
    #             exe, dirname=conf.pretrained, vars=load_vars)
    #         for var in load_vars:
    #             print("Parameter[{}] loaded sucessfully!".format(var.name))
            
    #         # not matched
    #         for var in load_fail_vars:
    #             if not var.name.endswith("_loc"):
    #                 print(
    #                 "Parameter[{}] don't exist or shape does not match current network, skip"
    #                 " to load it.".format(var.name))
    #         print("{}/{} pretrained parameters loaded successfully!".format(
    #             len(load_vars),
    #             len(load_vars) + len(load_fail_vars)))
    #     else:
    #         print(
    #         'Pretrained model dir {} not exists, training from scratch...'.
    #         format(conf.pretrained))
    """

   
