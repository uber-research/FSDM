# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import time


class _Config:
    def __init__(self):
        self._init_logging_handler()
        self.cuda_device = 0
        self.eos_m_token = 'EOS_M'
        self.beam_len_bonus = 0.5
        self.mode = 'unknown'
        self.m = 'FSDM'
        self.prev_z_method = 'none'

        self.seed = 0

    def init_handler(self, data):
        init_method = {
            'camrest': self._camrest_init,
            'kvret': self._kvret_init
        }
        init_method[data]()

    def _camrest_init(self):
        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate' # or  = 'concat'
        self.separate_enc = True
        self.vocab_size = 850
        self.embedding_size = 50
        self.hidden_size = 50
        self.split = (3, 1, 1)
        self.lr = 0.003
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab-camrest.pkl'
        self.vocab_emb = './vocab/emb-camrest.npy'
        self.data = './data/CamRest676/CamRest676.json'
        self.entity = './data/CamRest676/CamRestOTGY.json'
        self.db = './data/CamRest676/CamRestDB.json'
        self.glove_path = './data/glove/glove.6B.50d.txt' #please download glove and place under this path
        self.batch_size = 32
        self.z_length = 8 #maximum length for belief state
        self.inf_length = 3 #maximum length for informable slot value
        self.req_length = 4 #maximum length for requestable slot value
        self.degree_size = 5 #size of knowledge base quired result
        self.layer_num = 1 #layer number for GRU
        self.dropout_rate = 0.5
        self.epoch_num = 100  # triggered by early stop
        self.cuda = False #use cuda or not
        self.spv_proportion = 100
        self.max_ts = 40 #maximum response decoder steps
        self.early_stop_count = 3
        self.new_vocab = True #create a new vocabulary
        self.model_path = './models/camrest.pkl'
        self.result_path = './results/camrest.csv'
        self.teacher_force = 100
        self.beam_search = True
        self.beam_size = 10
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False
        self.emb_trainable = False #make the embedding layer trainiable or not
        self.num_head = 1 #number of uniquet belief decoders, '=1' means that share the belief decoder among all slots, '=3' means each slot has a distinct belief decoder

    def _kvret_init(self):
        self.prev_z_method = 'separate'
        self.separate_enc = True
        self.intent = 'all'
        self.vocab_size = 1414
        self.embedding_size = 50
        self.hidden_size = 50
        self.split = None
        self.lr = 0.003
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab-kvret.pkl'
        self.vocab_emb = './vocab/emb-kvret.npy'
        self.train = './data/kvret/kvret_train_public.json'
        self.dev = './data/kvret/kvret_dev_public.json'
        self.test = './data/kvret/kvret_test_public.json'
        self.entity = './data/kvret/kvret_entities.json'
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.batch_size = 8
        self.degree_size = 8
        self.z_length = 8
        self.inf_length = 5
        self.req_length = 7
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100
        self.cuda = False
        self.spv_proportion = 100
        self.alpha = 0.0
        self.max_ts = 40
        self.early_stop_count = 3
        self.new_vocab = True
        self.model_path = './models/kvret.pkl'
        self.result_path = './results/kvret.csv'
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False
        self.oov_proportion = 100
        self.emb_trainable = False
        self.num_head = 1 #number of unique belief decoders, '=1' means that share the belief decoder among all slots, '=10' means each slot has a distinct belief decoder

    def __str__(self):
        s = ''
        for k, v in self.__dict__.items():
            s += '{} : {}\n'.format(k, v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)


global_config = _Config()
