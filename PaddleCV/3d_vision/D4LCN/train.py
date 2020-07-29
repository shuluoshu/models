#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

# from data.data_utils import *

sys.path.append(os.getcwd())
import lib.core as core
from lib.rpn_util import *
from data.d4lcn_reader import D4lcnReader
import pdb

from easydict import EasyDict as edict
import paddle
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid import framework
import math
from lib.loss.rpn_3d import *
import time

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    """parse"""
    parser = argparse.ArgumentParser("D4LCN train script")
    parser.add_argument(
        "--use_data_parallel",  # TODO
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use data parallel mode to train the model."
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='ResNet50',
        help='backbone model to train, default ResNet50')
    parser.add_argument(
        '--conf',
        type=str,
        default='depth_guided_config',
        help='config')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset',
        help='dataset directory')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
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
    args = parser.parse_args()
    return args


def train():
    """main train"""
    args = parse_args()

    # print_arguments(args)
    # check whether the installed paddle is compiled with GPU
    # check_gpu(args.use_gpu)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    #assert args.backbone in ['ResNet34', 'ResNet50'], \
    #    "--backbone unsupported"

    # conf init
    conf = core.init_config(args.conf)
    paths = core.init_training_paths(args.conf, conf.result_dir, args.save_dir)
    tracker = edict()
    start_iter = 0
    start_time = time.time()

    # get reader and anchor
    d4lcn_reader = D4lcnReader(conf, args.data_dir)
    epoch = (conf.max_iter / (d4lcn_reader.len / conf.batch_size)) + 1
    train_reader = d4lcn_reader.get_reader(conf.batch_size, mode='train')
    # pdb.set_trace()
    generate_anchors(conf, d4lcn_reader.data['train'], paths.output)
    compute_bbox_stats(conf, d4lcn_reader.data['train'], paths.output)
    pickle_write(os.path.join(paths.output, 'conf.pkl'), conf)

    # show configuration
    #pretty = pretty_print('conf', conf)
    #logging.info(pretty)
    # # train
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)
    save_parameters = (not args.use_data_parallel) or (
           args.use_data_parallel and fluid.dygraph.parallel.Env().local_rank == 0)

    with fluid.dygraph.guard(place):
        if args.ce:
            print("ce mode")
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        # -----------------------------------------
        # network and loss
        # -----------------------------------------

        # training network
        train_model, optimizer = core.init_training_model(conf, paths.output, args.conf) #remove backbone
        #train_model, optimizer = core.init_training_model(conf, args.backbone, paths.output, args.conf)

        # setup loss
        criterion_det = RPN_3D_loss(conf)

        if args.use_data_parallel:
            train_model = fluid.dygraph.parallel.DataParallel(train_model, strategy)

        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)
        # # pretrain = fluid.load_program_state("pretrained_model/DenseNet121_pretrained")
        # #state_dict = train_model.state_dict()
        # #Convert
        # #TODO load pretrained model

        # if os.path.exists(conf.pretrained):
        #      print("load pretrain model from ", conf.pretrained)
        #      pretrained, _ = fluid.load_dygraph(conf.pretrained)
        #      train_model.base.set_dict(pretrained, use_structured_name=True)
        #     #train_model.set_dict(pretrained, use_structured_name=True)
        # dict = train_model.state_dict()
        # fluid.save_dygraph(dict, "paddle")

        total_batch_num = 0

        for epo in range(int(epoch)):

            total_loss = 0.0
            total_acc1 = 0.0

            total_sample = 0

            for batch_id, data in enumerate(train_reader()):

                # NOTE: used in benchmark
                # if args.max_iter and total_batch_num == args.max_iter:
                #     return
                batch_start = time.time()
                #pdb.set_trace()

                images = np.array(
                    [x[0].reshape(3, 512, 1760) for x in data]).astype('float32')
                depths = np.array(
                    [x[1].reshape(3, 512, 1760) for x in data]).astype('float32')
                imobjs = np.array([x[2] for x in data])
                # import pickle
                # images = pickle.load(open("images.pkl","rb"))
                # imobjs = pickle.load(open("imobjs.pkl","rb"))
                # img = to_variable(images)
                #  learning rate
                # cur_iter = epo*100 + batch_id # TODO next_iter
                # adjust_lr(conf, optimizer, cur_iter)

                if len(np.array([x[2] for x in data])) != conf.batch_size:
                    continue

                img = to_variable(images)
                depth = to_variable(depths)
                # label = to_variable(y_data)
                # label.stop_gradient = True

                cls, prob, bbox_2d, bbox_3d, feat_size = train_model(img, depth)

                # f=open('./cls_paddle.pkl','wb')
                # pickle.dump(cls.numpy(),f)
                # f.close()
                # f=open('./prob_paddle.pkl','wb')
                # pickle.dump(prob.numpy(),f)
                # f.close()
                # f=open('./bbox_2d_paddle.pkl','wb')
                # pickle.dump(bbox_2d.numpy(),f)
                # f.close()
                # f=open('./bbox_3d_paddle.pkl','wb')
                # pickle.dump(bbox_3d.numpy(),f)
                # f.close()

                # # loss
                det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)

                total_loss = det_loss
                stats = det_stats

                # backprop
                if total_loss > 0:
                    if args.use_data_parallel:
                        total_loss = train_model.scale_loss(total_loss)
                        total_loss.backward()
                        train_model.apply_collective_grads()
                    else:
                        total_loss.backward()

                    # batch skip, simulates larger batches by skipping gradient step
                    if (not 'batch_skip' in conf) or ((batch_id + 1) % conf.batch_skip) == 0:
                        optimizer.minimize(total_loss)
                        optimizer.clear_gradients()

                batch_end = time.time()
                train_batch_cost = batch_end - batch_start

                # keep track of stats
                compute_stats(tracker, stats)

                # -----------------------------------------
                # display
                # -----------------------------------------
                iteration = epo * (d4lcn_reader.len/conf.batch_size) + batch_id

                if iteration % conf.display == 0 and iteration > start_iter:
                    # log results
                     log_stats(tracker, iteration, start_time, start_iter, conf.max_iter)
                     print( "epoch %d | batch step %d | iter %d, batch cost: %.5f, loss %0.3f" % \
                            (epo, batch_id, iteration, train_batch_cost, total_loss.numpy()))

                     # reset tracker
                     tracker = edict()

                # snapshot, do_test

                if (iteration % conf.snapshot_iter == 0 and iteration > start_iter) and save_parameters:
                    fluid.save_dygraph(train_model.state_dict(),
                                        '{}/iter{}_params'.format(paths.weights, iteration))
                    fluid.save_dygraph(optimizer.state_dict(),
                                       '{}/iter{}_opt'.format(paths.weights, iteration))

                    #do test
                    if conf.do_test:
                        train_model.phase=  "eval"
                        train_model.eval()
                        results_path = os.path.join(paths.results, 'results_{}'.format((epo)))
                    if conf.test_protocol.lower() == 'kitti':
                        results_path = os.path.join(results_path, 'data')
                        mkdir_if_missing(results_path, delete_if_exist=True)
                        test_kitti_3d(conf.dataset_test, "validation", train_model, conf, results_path, paths.data)
                    train_model.phase = "train"
                    train_model.train()


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

    """
