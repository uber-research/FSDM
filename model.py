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
import torch
import random
import numpy as np
from config import global_config as cfg
from reader import CamRest676Reader, get_glove_matrix
from reader import KvretReader
from network import FSDM, cuda_
from torch.optim import Adam
from torch.autograd import Variable
from reader import pad_sequences
import argparse, time

from metric import CamRestEvaluator, KvretEvaluator
import logging


class Model:
    def __init__(self, dataset):
        reader_dict = {
            'camrest': CamRest676Reader,
            'kvret': KvretReader,
        }
        model_dict = {
            'FSDM': FSDM
        }
        evaluator_dict = {
            'camrest': CamRestEvaluator,
            'kvret': KvretEvaluator,
        }
        self.reader = reader_dict[dataset]()
        self.m = model_dict[cfg.m](embed_size=cfg.embedding_size,
                                   hidden_size=cfg.hidden_size,
                                   vocab_size=cfg.vocab_size,
                                   layer_num=cfg.layer_num,
                                   dropout_rate=cfg.dropout_rate,
                                   z_length=cfg.z_length,
                                   max_ts=cfg.max_ts,
                                   beam_search=cfg.beam_search,
                                   beam_size=cfg.beam_size,
                                   eos_token_idx=self.reader.vocab.encode('EOS_M'),
                                   vocab=self.reader.vocab,
                                   teacher_force=cfg.teacher_force,
                                   degree_size=cfg.degree_size,
                                   num_head=cfg.num_head,
                                   separate_enc=cfg.separate_enc)
        self.EV = evaluator_dict[dataset]  # evaluator class
        if cfg.cuda:
            self.m = self.m.cuda()
        self.base_epoch = -1

    def _to_onehot(self, encoded):
        _np = np.zeros((cfg.vocab_size, 1))
        for idx in encoded:
            _np[idx] = 1.
        return _np

    def _convert_batch(self, py_batch, prev_z_py=None):
        kw_ret = {}

        requested_7_np = np.stack(py_batch['requested_7'], axis=0).transpose()
        requested_7_np = requested_7_np[:, :, np.newaxis]  # 7, batchsize, 1
        response_7_np = np.stack(py_batch['response_7'], axis=0).transpose()
        response_7_np = response_7_np[:, :, np.newaxis]  # 7, batchsize, 1
        requestable_key = py_batch['requestable_key']  # (batchsize, 7) keys
        requestable_slot = py_batch['requestable_slot']  # (batchsize, 7) slots
        requestable_key_np = pad_sequences(requestable_key, len(requestable_key[0]), padding='post',
                                           truncating='post').transpose((1, 0))
        requestable_slot_np = pad_sequences(requestable_slot, len(requestable_slot[0]), padding='post',
                                            truncating='post').transpose((1, 0))
        kw_ret['requestable_key_np'] = requestable_key_np
        kw_ret['requestable_slot_np'] = requestable_slot_np
        kw_ret['requestable_key'] = cuda_(Variable(torch.from_numpy(requestable_key_np).long()))
        kw_ret['requestable_slot'] = cuda_(Variable(torch.from_numpy(requestable_slot_np).long()))
        kw_ret['requested_7'] = cuda_(Variable(torch.from_numpy(requested_7_np).float()))
        kw_ret['response_7'] = cuda_(Variable(torch.from_numpy(response_7_np).float()))

        u_input_py = py_batch['user']
        u_len_py = py_batch['u_len']

        if cfg.prev_z_method == 'concat' and prev_z_py is not None:
            for i in range(len(u_input_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    u_input_py[i] = prev_z_py[i][:idx + 1] + u_input_py[i]
                else:
                    u_input_py[i] = prev_z_py[i] + u_input_py[i]
                u_len_py[i] = len(u_input_py[i])
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2  # unk
        elif cfg.prev_z_method == 'separate' and prev_z_py is not None:
            for i in range(len(prev_z_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    prev_z_py[i] = prev_z_py[i][:idx + 1]
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2  # unk
            prev_z_input_np = pad_sequences(prev_z_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_len = np.array([len(_) for _ in prev_z_py])
            prev_z_input = cuda_(Variable(torch.from_numpy(prev_z_input_np).long()))
            kw_ret['prev_z_len'] = prev_z_len
            kw_ret['prev_z_input'] = prev_z_input
            kw_ret['prev_z_input_np'] = prev_z_input_np

        degree_input_np = np.array(py_batch['degree'])
        u_input_np = pad_sequences(u_input_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        m_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose(
            (1, 0))
        r_input_np = pad_sequences(py_batch['requested'], cfg.req_length, padding='post', truncating='post').transpose(
            (1, 0))  # (seqlen, batchsize)
        k_input_np = pad_sequences(py_batch['constraint_key'], len(py_batch['constraint_key'][0]), padding='post',
                                   truncating='post').transpose(
            (1, 0))
        flat_constraint_value = []
        num_k = k_input_np.shape[0]
        for b in py_batch['constraint_value']:
            for k in b:
                flat_constraint_value.append(k)

        flat_i_input_np = pad_sequences(flat_constraint_value, cfg.inf_length, padding='post', truncating='post')

        i_input_np = []
        i_k_input_np = []
        for idx, k in enumerate(flat_i_input_np):
            i_k_input_np.append(k)
            if (idx + 1) % num_k == 0:
                i_input_np.append(np.asarray(i_k_input_np))
                i_k_input_np = []
        i_input_np = np.asarray(i_input_np)  # (batchsize, key_size, seqlen)

        u_len = np.array(u_len_py)
        m_len = np.array(py_batch['m_len'])

        degree_input = cuda_(Variable(torch.from_numpy(degree_input_np).float()))
        u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
        m_input = cuda_(Variable(torch.from_numpy(m_input_np).long()))
        r_input = cuda_(Variable(torch.from_numpy(r_input_np).long()))
        k_input = cuda_(Variable(torch.from_numpy(k_input_np).long()))
        i_input = cuda_(Variable(torch.from_numpy(i_input_np).long()))
        i_input = i_input.permute(1, 2, 0)
        z_input = []
        for k_i_input in i_input:
            z_input.append(k_i_input)
        z_input = torch.cat(z_input, dim=0)
        z_input_np = z_input.cpu().data.numpy()

        kw_ret['z_input_np'] = z_input_np

        return u_input, u_input_np, z_input, m_input, m_input_np, u_len, m_len, \
               degree_input, k_input, i_input, r_input, kw_ret, py_batch['constraint_eos']

    def _test_convert_batch(self, py_batch, prev_z_py=None, prev_m_py=None):  # ???not easy to write
        kw_ret = {}

        requested_7_np = np.stack(py_batch['requested_7'], axis=0).transpose()
        requested_7_np = requested_7_np[:, :, np.newaxis]  # 7, batchsize, 1
        response_7_np = np.stack(py_batch['response_7'], axis=0).transpose()
        response_7_np = response_7_np[:, :, np.newaxis]  # 7, batchsize, 1
        requestable_key = py_batch['requestable_key']  # (batchsize, 7) keys
        requestable_slot = py_batch['requestable_slot']  # (batchsize, 7) slots
        requestable_key_np = pad_sequences(requestable_key, len(requestable_key[0]), padding='post',
                                           truncating='pre').transpose((1, 0))
        requestable_slot_np = pad_sequences(requestable_slot, len(requestable_slot[0]), padding='post',
                                            truncating='pre').transpose((1, 0))
        kw_ret['requestable_key_np'] = requestable_key_np
        kw_ret['requestable_slot_np'] = requestable_slot_np
        kw_ret['requestable_key'] = cuda_(Variable(torch.from_numpy(requestable_key_np).long()))
        kw_ret['requestable_slot'] = cuda_(Variable(torch.from_numpy(requestable_slot_np).long()))
        kw_ret['requested_7'] = cuda_(Variable(torch.from_numpy(requested_7_np).float()))
        kw_ret['response_7'] = cuda_(Variable(torch.from_numpy(response_7_np).float()))
        u_input_py = py_batch['user']
        u_len_py = py_batch['u_len']

        eom = self.reader.vocab.encode('EOS_M')
        if prev_m_py != None:
            fix_u_input_py = []
            for b, m in zip(u_input_py, prev_m_py):
                if eom in b:
                    idx = b.index(eom)
                    b = b[idx + 1:]
                if eom in m:
                    idx = m.index(eom)
                    m = m[:idx + 1]
                    m = [self.reader.vocab.encode('<unk>') if w >= cfg.vocab_size else w for w in m]
                    fix_u_input_py.append(m + b)
                else:
                    fix_u_input_py.append(b)
            u_input_py = fix_u_input_py
            u_len_py = [len(b) for b in fix_u_input_py]

        if cfg.prev_z_method == 'concat' and prev_z_py is not None:
            for i in range(len(u_input_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    u_input_py[i] = prev_z_py[i][:idx + 1] + u_input_py[i]
                else:
                    u_input_py[i] = prev_z_py[i] + u_input_py[i]
                u_len_py[i] = len(u_input_py[i])
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2  # unk
        elif cfg.prev_z_method == 'separate' and prev_z_py is not None:
            for i in range(len(prev_z_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    prev_z_py[i] = prev_z_py[i][:idx + 1]
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2  # unk
            prev_z_input_np = pad_sequences(prev_z_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_len = np.array([len(_) for _ in prev_z_py])
            prev_z_input = cuda_(Variable(torch.from_numpy(prev_z_input_np).long()))
            kw_ret['prev_z_len'] = prev_z_len
            kw_ret['prev_z_input'] = prev_z_input
            kw_ret['prev_z_input_np'] = prev_z_input_np

        degree_input_np = np.array(py_batch['degree'])
        u_input_np = pad_sequences(u_input_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        m_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose(
            (1, 0))

        r_input_np = pad_sequences(py_batch['requested'], cfg.req_length, padding='post', truncating='post').transpose(
            (1, 0))  # (seqlen, batchsize)
        k_input_np = pad_sequences(py_batch['constraint_key'], len(py_batch['constraint_key'][0]), padding='post',
                                   truncating='post').transpose(
            (1, 0))
        flat_constraint_value = []
        num_k = k_input_np.shape[0]
        for b in py_batch['constraint_value']:
            for k in b:
                flat_constraint_value.append(k)
        inf_length = max([len(l) for l in flat_constraint_value])
        print(inf_length)
        flat_i_input_np = pad_sequences(flat_constraint_value, cfg.inf_length, padding='post', truncating='post')

        i_input_np = []
        i_k_input_np = []
        for idx, k in enumerate(flat_i_input_np):
            i_k_input_np.append(k)
            if (idx + 1) % num_k == 0:
                i_input_np.append(np.asarray(i_k_input_np))
                i_k_input_np = []
        i_input_np = np.asarray(i_input_np)  # (batchsize, key_size, seqlen)

        u_len = np.array(u_len_py)
        m_len = np.array(py_batch['m_len'])

        degree_input = cuda_(Variable(torch.from_numpy(degree_input_np).float()))
        u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
        m_input = cuda_(Variable(torch.from_numpy(m_input_np).long()))
        r_input = cuda_(Variable(torch.from_numpy(r_input_np).long()))
        k_input = cuda_(Variable(torch.from_numpy(k_input_np).long()))
        i_input = cuda_(Variable(torch.from_numpy(i_input_np).long()))
        i_input = i_input.permute(1, 2, 0)
        z_input = []
        for k_i_input in i_input:
            z_input.append(k_i_input)
        z_input = torch.cat(z_input, dim=0)
        z_input_np = z_input.cpu().data.numpy()

        kw_ret['z_input_np'] = z_input_np

        if 'database' in py_batch.keys():
            database = py_batch['database']
        else:
            database = None

        return u_input, u_input_np, z_input, m_input, m_input_np, u_len, m_len, \
               degree_input, k_input, i_input, r_input, kw_ret, database, py_batch['constraint_eos']

    def train(self):
        lr = cfg.lr
        prev_min_loss = 0.
        early_stop_count = cfg.early_stop_count
        train_time = 0
        for epoch in range(cfg.epoch_num):
            loss_weights = [1., 1., 1., 1.]
            sw = time.time()
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            self.m.self_adjust(epoch)
            sup_loss = 0
            sup_cnt = 0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=1e-5)
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    if cfg.truncated:
                        logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()
                    u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                    m_len, degree_input, k_input, i_input, r_input, kw_ret, constraint_eos \
                        = self._convert_batch(turn_batch, prev_z)

                    loss, pr_loss, m_loss, turn_states, req_loss, res_loss = self.m(u_input=u_input, z_input=z_input,
                                                                                    m_input=m_input,
                                                                                    degree_input=degree_input,
                                                                                    u_input_np=u_input_np,
                                                                                    m_input_np=m_input_np,
                                                                                    turn_states=turn_states,
                                                                                    u_len=u_len, m_len=m_len,
                                                                                    k_input=k_input,
                                                                                    i_input=i_input,
                                                                                    r_input=r_input,
                                                                                    loss_weights=loss_weights,
                                                                                    mode='train', **kw_ret)
                    loss.backward(retain_graph=turn_num != len(dial_batch) - 1)
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 10.0)
                    optim.step()
                    sup_loss += loss.cpu().item()
                    sup_cnt += 1
                    logging.debug(
                        'loss:{} pr_loss:{} req_loss:{} res_loss:{}  m_loss:{} grad:{}'.format(loss.cpu().item(),
                                                                                               pr_loss.cpu().item(),
                                                                                               req_loss.cpu().item(),
                                                                                               res_loss.cpu().item(),
                                                                                               m_loss.cpu().item(),
                                                                                               grad))

                    prev_z = turn_batch['bspan']

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            train_time += time.time() - sw
            logging.info('Traning time: {}'.format(train_time))
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            logging.info('time for epoch %d: %f' % (epoch, time.time() - sw))
            valid_loss = valid_sup_loss + valid_unsup_loss

            metrics = self.eval(data='dev')
            valid_metrics = metrics[-1] + metrics[-2] + metrics[-3]
            logging.info('valid metric %f ' % (valid_metrics))
            if valid_metrics >= prev_min_loss:
                self.save_model(epoch)
                prev_min_loss = valid_metrics
                early_stop_count = cfg.early_stop_count
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def eval(self, data='test'):
        self.m.eval()
        self.reader.result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test' if not cfg.pretrain else 'pretrain_test'
        for batch_num, dial_batch in enumerate(data_iterator):
            turn_states = {}
            prev_z = None
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, k_input, i_input, r_input, kw_ret, constraint_eos \
                    = self._convert_batch(turn_batch, prev_z)

                m_idx, z_idx, turn_states = self.m(u_input=u_input, z_input=z_input,
                                                            m_input=m_input,
                                                            degree_input=degree_input,
                                                            u_input_np=u_input_np,
                                                            m_input_np=m_input_np,
                                                            turn_states=turn_states,
                                                            u_len=u_len, m_len=m_len,
                                                            k_input=k_input,
                                                            i_input=i_input,
                                                            r_input=r_input,
                                                            mode='test', **kw_ret)
                self.reader.wrap_result(turn_batch, m_idx, z_idx, prev_z=prev_z)
                prev_z = z_idx
        if self.reader.result_file != None:
            self.reader.result_file.close()
        ev = self.EV(result_path=cfg.result_path, data=data)
        res = ev.run_metrics()
        self.m.train()
        return res


    def validate(self, loss_weights=[1., 1., 1., 1.], data='dev'):
        self.m.eval()
        data_iterator = self.reader.mini_batch_iterator(data)
        sup_loss, unsup_loss = 0, 0
        sup_cnt, unsup_cnt = 0, 0
        for dial_batch in data_iterator:
            turn_states = {}
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, k_input, i_input, r_input, kw_ret, constraint_eos \
                    = self._convert_batch(turn_batch)

                loss, pr_loss, m_loss, turn_states, req_loss, res_loss = self.m(u_input=u_input, z_input=z_input,
                                                                                m_input=m_input,
                                                                                degree_input=degree_input,
                                                                                u_input_np=u_input_np,
                                                                                m_input_np=m_input_np,
                                                                                turn_states=turn_states,
                                                                                u_len=u_len, m_len=m_len,
                                                                                k_input=k_input,
                                                                                i_input=i_input,
                                                                                r_input=r_input,
                                                                                loss_weights=loss_weights,
                                                                                mode='train', **kw_ret)
                sup_loss += loss.cpu().item()
                sup_cnt += 1
                logging.debug(
                    'loss:{} pr_loss:{} req_loss:{} res_loss:{} m_loss:{}'.format(loss.cpu().item(), pr_loss.cpu().item(),
                                                                                  req_loss.cpu().item(),
                                                                                  res_loss.cpu().item(),
                                                                                  m_loss.cpu().item()))

        sup_loss /= (sup_cnt + 1e-8)
        unsup_loss /= (unsup_cnt + 1e-8)
        self.m.train()
        return sup_loss, unsup_loss

    def save_model(self, epoch, path=None):
        if not path:
            path = cfg.model_path
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path)
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self):
        initial_arr = self.m.u_encoder.embedding.weight.data.cpu().numpy()
        embedding_arr = torch.from_numpy(get_glove_matrix(self.reader.vocab, initial_arr))

        self.m.u_encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.u_encoder.embedding.weight.requires_grad = cfg.emb_trainable
        if cfg.separate_enc:
            self.m.z_encoder.embedding.weight.data.copy_(embedding_arr)
            self.m.z_encoder.embedding.weight.requires_grad = cfg.emb_trainable
        for i in range(cfg.num_head):
            self.m.z_decoders[i].emb.weight.data.copy_(embedding_arr)
            self.m.z_decoders[i].emb.weight.requires_grad = cfg.emb_trainable
        self.m.req_classifiers.emb.weight.data.copy_(embedding_arr)
        self.m.req_classifiers.emb.weight.requires_grad = cfg.emb_trainable
        self.m.res_classifiers.emb.weight.data.copy_(embedding_arr)
        self.m.res_classifiers.emb.weight.requires_grad = cfg.emb_trainable
        self.m.m_decoder.emb.weight.data.copy_(embedding_arr)
        self.m.m_decoder.emb.weight.requires_grad = cfg.emb_trainable

    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters if p.requires_grad == True])
        print('total trainable params: %d' % param_cnt)
        print(self.m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-data')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()
    cfg.init_handler(args.data)

    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            else:
                v = dtype(v)
            setattr(cfg, k, v)

    logging.debug(str(cfg))
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.debug('Device: {}'.format(torch.cuda.current_device()))
    cfg.mode = args.mode

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    m = Model(args.data.split('-')[-1])
    m.count_params()
    if args.mode == 'train':
        m.load_glove_embedding()
        m.m.beam_search = False
        m.train()
    elif args.mode == 'adjust':
        m.load_model()
        m.train()
        m.load_model()
        m.eval()
    elif args.mode == 'test':
        m.load_model()
        m.eval(data='test')


if __name__ == '__main__':
    main()
