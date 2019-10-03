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
import numpy as np
import json
import pickle
from config import global_config as cfg
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
import random
import os
import re
import csv


def clean_replace(s, r, t, forward=True, backward=False):
    def clean_replace_single(s, r, t, forward, backward, sidx=0):
        idx = s[sidx:].find(r)
        if idx == -1:
            return s, -1
        idx += sidx
        idx_r = idx + len(r)
        if backward:
            while idx > 0 and s[idx - 1]:
                idx -= 1
        elif idx > 0 and s[idx - 1] != ' ':
            return s, -1

        if forward:
            while idx_r < len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                idx_r += 1
        elif idx_r != len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
            return s, -1
        return s[:idx] + t + s[idx_r:], idx_r

    sidx = 0
    while sidx != -1:
        s, sidx = clean_replace_single(s, r, t, forward, backward, sidx)
    return s


class _ReaderBase:
    class LabelSet:
        def __init__(self):
            self._idx2item = {}
            self._item2idx = {}
            self._freq_dict = {}

        def __len__(self):
            return len(self._idx2item)

        def _absolute_add_item(self, item):
            idx = len(self)
            self._idx2item[idx] = item
            self._item2idx[item] = idx

        def add_item(self, item):
            if item not in self._freq_dict:
                self._freq_dict[item] = 0
            self._freq_dict[item] += 1

        def construct(self, limit):
            l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
            print('Actual label size %d' % (len(l) + len(self._idx2item)))
            if len(l) + len(self._idx2item) < limit:
                logging.warning('actual label set smaller than that configured: {}/{}'
                                .format(len(l) + len(self._idx2item), limit))
            for item in l:
                if item not in self._item2idx:
                    idx = len(self._idx2item)
                    self._idx2item[idx] = item
                    self._item2idx[item] = idx
                    if len(self._idx2item) >= limit:
                        break

        def encode(self, item):
            return self._item2idx[item]

        def decode(self, idx):
            return self._idx2item[idx]

    class Vocab(LabelSet):
        def __init__(self, init=True):
            _ReaderBase.LabelSet.__init__(self)
            if init:
                self._absolute_add_item('<pad>')  # 0
                self._absolute_add_item('<go>')  # 1
                self._absolute_add_item('<unk>')  # 2
                self._absolute_add_item('<go2>')  # 3
                self._absolute_add_item('<goReq>')  # 4

        def load_vocab(self, vocab_path):
            f = open(vocab_path, 'rb')
            dic = pickle.load(f)
            self._idx2item = dic['idx2item']
            self._item2idx = dic['item2idx']
            self._freq_dict = dic['freq_dict']
            f.close()

        def save_vocab(self, vocab_path):
            f = open(vocab_path, 'wb')
            dic = {
                'idx2item': self._idx2item,
                'item2idx': self._item2idx,
                'freq_dict': self._freq_dict
            }
            pickle.dump(dic, f)
            f.close()

        def sentence_encode(self, word_list):
            return [self.encode(_) for _ in word_list]

        def sentence_decode(self, index_list, eos=None):
            l = [self.decode(_) for _ in index_list]
            if not eos or eos not in l:
                return ' '.join(l)
            else:
                idx = l.index(eos)
                return ' '.join(l[:idx])

        def nl_decode(self, l, eos=None):
            return [self.sentence_decode(_, eos) + '\n' for _ in l]

        def encode(self, item):
            if item in self._item2idx:
                return self._item2idx[item]
            else:
                return self._item2idx['<unk>']

        def decode(self, idx):
            if idx < len(self):
                return self._idx2item[idx]
            else:
                return 'ITEM_%d' % (idx - cfg.vocab_size)

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = self.Vocab()
        self.result_file = ''

    def _construct(self, *args):
        """
        load data, construct vocab and store them in self.train/dev/test
        :param args:
        :return:
        """
        raise NotImplementedError('This is an abstract class, bro')

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5: del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return turn_bucket

    def _mark_batch_as_supervised(self, all_batches):
        supervised_num = int(len(all_batches) * cfg.spv_proportion / 100)
        for i, batch in enumerate(all_batches):
            for dial in batch:
                for turn in dial:
                    turn['supervised'] = i < supervised_num
                    if not turn['supervised']:
                        turn['degree'] = [0.] * cfg.degree_size  # unsupervised learning. DB degree should be unknown
        return all_batches

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == cfg.batch_size:
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def _transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def mini_batch_iterator(self, set_name):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []
        for k in turn_bucket:
            batches = self._construct_mini_batch(turn_bucket[k])
            all_batches += batches
        self._mark_batch_as_supervised(all_batches)
        random.shuffle(all_batches)
        for i, batch in enumerate(all_batches):
            yield self._transpose_batch(batch)

    def wrap_result(self, turn_batch, gen_m, gen_z, eos_syntax=None, prev_z=None):
        """
        wrap generated results
        :param gen_z:
        :param gen_m:
        :param turn_batch: dict of [i_1,i_2,...,i_b] with keys
        :return:
        """

        results = []
        if eos_syntax is None:
            eos_syntax = {'response': 'EOS_M', 'user': 'EOS_U', 'bspan': 'EOS_Z2'}
        batch_size = len(turn_batch['user'])
        for i in range(batch_size):
            entry = {}
            if prev_z is not None:
                src = prev_z[i] + turn_batch['user'][i]
            else:
                src = turn_batch['user'][i]
            for key in turn_batch:
                entry[key] = turn_batch[key][i]
                if key in eos_syntax:
                    entry[key] = self.vocab.sentence_decode(entry[key], eos=eos_syntax[key])
            if gen_m:
                entry['generated_response'] = self.vocab.sentence_decode(gen_m[i], eos='EOS_M')
            else:
                entry['generated_response'] = ''
            if gen_z:
                entry['generated_bspan'] = self.vocab.sentence_decode( gen_z[i], eos='EOS_Z2')
            else:
                entry['generated_bspan'] = ''
            results.append(entry)
        write_header = False
        if not self.result_file:
            self.result_file = open(cfg.result_path, 'w')
            self.result_file.write(str(cfg))
            write_header = True

        field = ['dial_id', 'turn_num', 'user', 'generated_bspan', 'bspan', 'generated_response', 'response', 'u_len',
                 'm_len', 'supervised']
        for result in results:
            del_k = []
            for k in result:
                if k not in field:
                    del_k.append(k)
            for k in del_k:
                result.pop(k)
        writer = csv.DictWriter(self.result_file, fieldnames=field)
        if write_header:
            self.result_file.write('START_CSV_SECTION\n')
            writer.writeheader()
        writer.writerows(results)
        return results

    def db_search(self, constraints):
        raise NotImplementedError('This is an abstract method')

    def db_degree_handler(self, z_samples):
        """
        returns degree of database searching and it may be used to control further decoding.
        One hot vector, indicating the number of entries found: [0, 1, 2, 3, 4, >=5]
        :param z_samples: nested list of B * [T]
        :return: an one-hot control *numpy* control vector
        """
        control_vec = []

        for cons_idx_list in z_samples:
            constraints = set()
            for cons in cons_idx_list:
                cons = self.vocab.decode(cons)
                if cons == 'EOS_Z1':
                    break
                constraints.add(cons)
            match_result = self.db_search(constraints)
            degree = len(match_result)
            # modified
            # degree = 0
            control_vec.append(self._degree_vec_mapping(degree))
        return np.array(control_vec)

    def _degree_vec_mapping(self, match_num):
        l = [0.] * cfg.degree_size
        l[min(cfg.degree_size - 1, match_num)] = 1.
        return l


class CamRest676Reader(_ReaderBase):
    def __init__(self):
        super().__init__()
        self._construct(cfg.data, cfg.db)
        self.result_file = ''
        self.db = []

    def _get_tokenized_data(self, raw_data, db_data, construct_vocab):
        tokenized_data = []
        vk_map = self._value_key_map(db_data)
        for dial_id, dial in enumerate(raw_data):
            tokenized_dial = []
            for turn in dial['dial']:
                turn_num = turn['turn']
                constraint = []
                constraint_dict = {'food': ['EOS_food'], 'pricerange': ['EOS_pricerange'], 'area': ['EOS_area']}
                constraint_eos = [v[0] for v in constraint_dict.values()]
                requested = []
                for slot in turn['usr']['slu']:
                    if slot['act'] == 'inform':
                        s = slot['slots'][0][1]
                        if slot['slots'][0][0] in constraint_dict:
                            constraint_dict[slot['slots'][0][0]] = word_tokenize(s) + ['EOS_' + slot['slots'][0][0]]
                        else:
                            print('something wrong here: ', s)
                            assert False
                        if s not in ['dontcare', 'none']:
                            constraint.extend(word_tokenize(s))
                    else:
                        requested.extend(word_tokenize(slot['slots'][0][1]))
                degree = len(self.db_search(constraint))
                requested = sorted(requested)
                constraint.append('EOS_Z1')
                requested.append('EOS_Z2')
                constraint_key = list(constraint_dict.keys())
                constraint_value = list(constraint_dict.values())
                user = word_tokenize(turn['usr']['transcript']) + ['EOS_U']
                response = word_tokenize(self._replace_entity(turn['sys']['sent'], vk_map, constraint)) + ['EOS_M']
                tokenized_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': user,
                    'response': response,
                    'constraint': constraint,
                    'requested': requested,
                    'degree': degree,
                    'constraint_key': constraint_key,
                    'constraint_value': constraint_value,
                    'constraint_eos': constraint_eos,
                })
                if construct_vocab:
                    for word in user + response + constraint + requested + constraint_key:
                        self.vocab.add_item(word)
                    for list_word in constraint_value:
                        for word in list_word:
                            self.vocab.add_item(word)
            tokenized_data.append(tokenized_dial)
        return tokenized_data

    def _replace_entity(self, response, vk_map, constraint):
        response = re.sub('[cC][., ]*[bB][., ]*\d[., ]*\d[., ]*\w[., ]*\w', 'postcode_SLOT', response)
        response = re.sub('\d{5}\s?\d{6}', 'phone_SLOT', response)
        constraint_str = ' '.join(constraint)
        for v, k in sorted(vk_map.items(), key=lambda x: -len(x[0])):
            start_idx = response.find(v)
            if start_idx == -1 \
                    or (start_idx != 0 and response[start_idx - 1] != ' ') \
                    or (v in constraint_str):
                continue
            if k not in ['name', 'address']:
                response = clean_replace(response, v, k + '_SLOT', forward=True, backward=False)
            else:
                response = clean_replace(response, v, k + '_SLOT', forward=False, backward=False)
        return response

    def _value_key_map(self, db_data):
        requestable_keys = ['address', 'name', 'phone', 'postcode', 'food', 'area', 'pricerange']
        value_key = {}
        for db_entry in db_data:
            for k, v in db_entry.items():
                if k in requestable_keys:
                    value_key[v] = k
        return value_key

    def _get_encoded_data(self, tokenized_data):
        requestable_keys = ['address', 'name', 'phone', 'postcode', 'food', 'area', 'pricerange']
        requestable_slots = [k + '_SLOT' for k in requestable_keys]
        requestable_keys = self.vocab.sentence_encode(requestable_keys)
        requestable_slots = self.vocab.sentence_encode(requestable_slots)
        encoded_data = []
        for dial in tokenized_data:
            encoded_dial = []
            prev_response = []
            for turn in dial:
                user = self.vocab.sentence_encode(turn['user'])
                response = self.vocab.sentence_encode(turn['response'])
                constraint = self.vocab.sentence_encode(turn['constraint'])
                requested = self.vocab.sentence_encode(turn['requested'])
                degree = self._degree_vec_mapping(turn['degree'])
                turn_num = turn['turn_num']
                dial_id = turn['dial_id']
                constraint_key = self.vocab.sentence_encode(turn['constraint_key'])
                constraint_value = [self.vocab.sentence_encode(v) for v in turn['constraint_value']]
                constraint_eos = self.vocab.sentence_encode(turn['constraint_eos'])
                requested_7 = np.asarray([1 if r in requested else 0 for r in requestable_keys])
                response_7 = np.asarray([1 if r in response else 0 for r in requestable_slots])
                # final input
                encoded_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': prev_response + user,
                    'response': response,
                    'bspan': [w for c in constraint_value for w in c] + requested,  # constraint + requested,
                    'u_len': len(prev_response + user),
                    'm_len': len(response),
                    'degree': degree,
                    'constraint_key': constraint_key,
                    'constraint_value': constraint_value,
                    'requested': requested,
                    'constraint_eos': constraint_eos,
                    'requestable_key': requestable_keys,
                    'requestable_slot': requestable_slots,
                    'requested_7': requested_7,
                    'response_7': response_7,
                })
                # modified
                prev_response = response
            encoded_data.append(encoded_dial)
        return encoded_data

    def _split_data(self, encoded_data, split):
        """
        split data into train/dev/test
        :param encoded_data: list
        :param split: tuple / list
        :return:
        """
        total = sum(split)
        dev_thr = len(encoded_data) * split[0] // total
        test_thr = len(encoded_data) * (split[0] + split[1]) // total
        train, dev, test = encoded_data[:dev_thr], encoded_data[dev_thr:test_thr], encoded_data[test_thr:]
        return train, dev, test

    def _construct(self, data_json_path, db_json_path):
        """
        construct encoded train, dev, test set.
        :param data_json_path:
        :param db_json_path:
        :return:
        """
        construct_vocab = False
        if not os.path.isfile(cfg.vocab_path):
            construct_vocab = True
            print('Constructing vocab file...')
        raw_data_json = open(data_json_path)
        raw_data = json.loads(raw_data_json.read().lower())
        db_json = open(db_json_path)
        db_data = json.loads(db_json.read().lower())
        self.db = db_data
        tokenized_data = self._get_tokenized_data(raw_data, db_data, construct_vocab)
        if construct_vocab:
            self.vocab.construct(cfg.vocab_size)
            self.vocab.save_vocab(cfg.vocab_path)
        else:
            self.vocab.load_vocab(cfg.vocab_path)
        encoded_data = self._get_encoded_data(tokenized_data)
        self.train, self.dev, self.test = self._split_data(encoded_data, cfg.split)
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)
        raw_data_json.close()
        db_json.close()

    def db_search(self, constraints):
        match_results = []
        for entry in self.db:
            entry_values = ' '.join(entry.values())
            match = True
            for c in constraints:
                if c not in entry_values:
                    match = False
                    break
            if match:
                match_results.append(entry)
        return match_results


class KvretReader(_ReaderBase):
    def __init__(self):
        super().__init__()


        self.entity_dict = {}
        self.abbr_dict = {}

        self.wn = WordNetLemmatizer()

        self.tokenized_data_path = './data/kvret/'
        self._construct(cfg.train, cfg.dev, cfg.test, cfg.entity)

    def _construct(self, train_json_path, dev_json_path, test_json_path, entity_json_path):
        construct_vocab = True  # always construct vocab
        if not os.path.isfile(cfg.vocab_path):
            construct_vocab = True
            print('Constructing vocab file...')
        train_json, dev_json, test_json = open(train_json_path), open(dev_json_path), open(test_json_path)
        entity_json = open(entity_json_path)
        train_data, dev_data, test_data = json.loads(train_json.read().lower()), json.loads(dev_json.read().lower()), \
                                          json.loads(test_json.read().lower())
        entity_data = json.loads(entity_json.read().lower())
        self._get_entity_dict(entity_data)

        tokenized_train = self._get_tokenized_data(train_data, construct_vocab, 'train')
        tokenized_dev = self._get_tokenized_data(dev_data, construct_vocab, 'dev')
        tokenized_test = self._get_tokenized_data(test_data, construct_vocab, 'test')

        if construct_vocab:
            self.vocab.construct(cfg.vocab_size)
            self.vocab.save_vocab(cfg.vocab_path)
        else:
            self.vocab.load_vocab(cfg.vocab_path)

        self.train, self.dev, self.test = map(self._get_encoded_data, [tokenized_train, tokenized_dev,
                                                                       tokenized_test])
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)

    def _save_tokenized_data(self, data, filename):
        path = self.tokenized_data_path + filename + '.tokenized.json'
        f = open(path, 'w')
        json.dump(data, f, indent=2)
        f.close()

    def _load_tokenized_data(self, filename):
        '''
        path = self.tokenized_data_path + filename + '.tokenized.json'
        try:
            f = open(path,'r')
        except FileNotFoundError:
            return None
        data = json.load(f)
        f.close()
        return data
        '''
        return None

    def _tokenize(self, sent):
        return ' '.join(word_tokenize(sent))

    def _lemmatize(self, sent):
        return ' '.join([self.wn.lemmatize(_) for _ in sent.split()])

    def _replace_entity(self, response, vk_map, prev_user_input, intent):
        response = re.sub('\d+-?\d*fs?', 'temperature_SLOT', response)
        response = re.sub('\d+\s?miles?', 'distance_SLOT', response)
        response = re.sub('\d+\s\w+\s(dr)?(ct)?(rd)?(road)?(st)?(ave)?(way)?(pl)?\w*[.]?', 'address_SLOT', response)
        response = self._lemmatize(self._tokenize(response))
        requestable = {
            'weather': ['weather_attribute'],
            'navigate': ['poi', 'traffic', 'address', 'distance'],
            'schedule': ['event', 'date', 'time', 'party', 'agenda', 'room']
        }
        reqs = set()
        for v, k in sorted(vk_map.items(), key=lambda x: -len(x[0])):
            start_idx = response.find(v)
            if start_idx == -1 or k not in requestable[intent]:
                continue
            end_idx = start_idx + len(v)
            while end_idx < len(response) and response[end_idx] != ' ':
                end_idx += 1
            # test whether they are indeed the same word
            lm1, lm2 = v.replace('.', '').replace(' ', '').replace("'", ''), \
                       response[start_idx:end_idx].replace('.', '').replace(' ', '').replace("'", '')
            if lm1 == lm2 and lm1 not in prev_user_input and v not in prev_user_input:
                response = clean_replace(response, response[start_idx:end_idx], k + '_SLOT')
                reqs.add(k)
        return response, reqs

    def _clean_constraint_dict(self, constraint_dict, intent, prefer='short'):
        """
        clean the constraint dict so that every key is in "informable" and similar to one in provided entity dict.
        :param constraint_dict:
        :return:
        """
        informable = {
            'weather': ['date', 'location', 'weather_attribute'],
            'navigate': ['poi_type', 'distance'],
            'schedule': ['event', 'date', 'time', 'agenda', 'party', 'room']
        }

        del_key = set(constraint_dict.keys()).difference(informable[intent])
        for key in del_key:
            constraint_dict.pop(key)
        invalid_key = []
        for k in constraint_dict:
            constraint_dict[k] = constraint_dict[k].strip()
            v = self._lemmatize(self._tokenize(constraint_dict[k]))
            v = re.sub('(\d+) ([ap]m)', lambda x: x.group(1) + x.group(2), v)
            v = re.sub('(\d+)\s?(mile)s?', lambda x: x.group(1) + ' ' + x.group(2), v)
            if v in self.entity_dict:
                if prefer == 'short':
                    constraint_dict[k] = v
                elif prefer == 'long':
                    constraint_dict[k] = self.abbr_dict.get(v, v)
            elif v.split()[0] in self.entity_dict:
                if prefer == 'short':
                    constraint_dict[k] = v.split()[0]
                elif prefer == 'long':
                    constraint_dict[k] = self.abbr_dict.get(v.split()[0], v)
            else:
                invalid_key.append(k)

        ori_constraint_dict = constraint_dict.copy()
        for key in invalid_key:
            constraint_dict.pop(key)
        return constraint_dict, ori_constraint_dict

    def _get_tokenized_data(self, raw_data, add_to_vocab, data_type, is_test=False):
        """
        Somerrthing to note: We define requestable and informable slots as below in further experiments
        (including other baselines):

        informable = {
            'weather': ['date','location','weather_attribute'],
            'navigate': ['poi_type','distance'],
            'schedule': ['event']
        }

        requestable = {
            'weather': ['weather_attribute'],
            'navigate': ['poi','traffic','address','distance'],
            'schedule': ['event','date','time','party','agenda','room']
        }
        :param raw_data:
        :param add_to_vocab:
        :param data_type:
        :return:
        """
        requestable = {
            'weather': ['weather_attribute'],
            'navigate': ['poi', 'traffic', 'address', 'distance'],
            'schedule': ['date', 'time', 'party', 'agenda', 'room']
        }
        informable_keys = ['date', 'location', 'weather_attribute', 'poi_type', 'distance', 'event', 'time',
                           'agenda', 'party', 'room']  # !!!2 date
        requestable_keys = ['weather_attribute', 'temperature', 'poi', 'traffic', 'address', 'distance', 'event',
                            'date', 'time',
                            'party', 'agenda', 'room']
        tokenized_data = self._load_tokenized_data(data_type)
        if tokenized_data is not None:
            logging.info('directly loading %s' % data_type)
            return tokenized_data
        tokenized_data = []
        state_dump = {}
        for dial_id, raw_dial in enumerate(raw_data):
            tokenized_dial = []
            prev_utter = ''
            single_turn = {}
            constraint_dict = {}
            intent = raw_dial['scenario']['task']['intent']
            db = raw_dial['scenario']['kb']['items']
            if cfg.intent != 'all' and cfg.intent != intent:
                if intent not in ['navigate', 'weather', 'schedule']:
                    raise ValueError('what is %s intent bro?' % intent)
                else:
                    continue
            prev_response = []
            for turn_num, dial_turn in enumerate(raw_dial['dialogue']):
                state_dump[(dial_id, turn_num)] = {}
                if dial_turn['turn'] == 'driver':
                    u = self._lemmatize(self._tokenize(dial_turn['data']['utterance']))
                    u = re.sub('(\d+) ([ap]m)', lambda x: x.group(1) + x.group(2), u)
                    single_turn['user'] = prev_response + u.split() + ['EOS_U']
                    prev_utter += ' ' + u
                elif dial_turn['turn'] == 'assistant':
                    s = dial_turn['data']['utterance']
                    # find entities and replace them
                    s = re.sub('(\d+) ([ap]m)', lambda x: x.group(1) + x.group(2), s)
                    s, reqs = self._replace_entity(s, self.entity_dict, prev_utter, intent)
                    single_turn['response'] = s.split() + ['EOS_M']
                    # get constraints
                    if not constraint_dict:
                        constraint_dict = dial_turn['data']['slots']
                    else:
                        for k, v in dial_turn['data']['slots'].items():
                            constraint_dict[k] = v

                    slots_key = set(list(constraint_dict.keys()))
                    constraint_dict, ori_constraint_dict = self._clean_constraint_dict(constraint_dict, intent)
                    #!!!!!!mark clean_constraint eg, distance: nearest is cleaned due to nearest

                    raw_constraints = constraint_dict.values()
                    raw_constraints = [self._lemmatize(self._tokenize(_)) for _ in raw_constraints]

                    Lei_constraint_dict = {k: (self._lemmatize(
                        self._tokenize(constraint_dict[k])) + ' EOS_' + k).split() if k in constraint_dict else [
                        'EOS_' + k]
                                           for k in informable_keys}
                    Lei_response_key = [k + '_SLOT' for k in requestable_keys]

                    # add separator
                    constraints = []
                    for item in raw_constraints:
                        if constraints:
                            constraints.append(';')
                        constraints.extend(item.split())
                    # get requests
                    dataset_requested = set(
                        filter(lambda x: dial_turn['data']['requested'][x], dial_turn['data']['requested'].keys()))

                    requests = sorted(list(dataset_requested.intersection(reqs)))
                    # requests = sorted(list(dataset_requested - slots_key))
                    requested_7 = [1 if r in requests else 0 for r in requestable_keys]
                    response_slot = sorted(set(single_turn['response']).intersection(set(Lei_response_key)))
                    response_7 = [1 if r in response_slot else 0 for r in Lei_response_key]

                    single_turn['constraint'] = constraints + ['EOS_Z1']
                    single_turn['requested'] = requests + ['EOS_Z2']
                    single_turn['turn_num'] = len(tokenized_dial)
                    single_turn['dial_id'] = dial_id
                    single_turn['database'] = raw_dial['scenario']['kb']['items']
                    single_turn['degree'] = self.Lei_db_degree(constraint_dict, ori_constraint_dict,
                                                               raw_dial['scenario']['kb']['items'], intent)
                    single_turn['constraint_key'] = list(Lei_constraint_dict.keys())
                    single_turn['constraint_value'] = list(Lei_constraint_dict.values())
                    single_turn['requestable_key'] = requestable_keys
                    single_turn['requestable_slot'] = Lei_response_key
                    single_turn['requested_7'] = requested_7
                    single_turn['response_7'] = response_7
                    if 'user' in single_turn:
                        state_dump[(dial_id, len(tokenized_dial))]['constraint'] = constraint_dict
                        state_dump[(dial_id, len(tokenized_dial))]['request'] = requests
                        tokenized_dial.append(single_turn)
                    prev_response = single_turn['response']
                    single_turn = {}
                    # print(constraints)
                    #print(list(Lei_constraint_dict.values()))
            if add_to_vocab:
                for single_turn in tokenized_dial:
                    for word_token in single_turn['constraint'] + single_turn['requested'] + \
                                      single_turn['constraint_key'] + \
                                      single_turn['requestable_key'] + single_turn['requestable_slot'] + \
                                      single_turn['user'] + single_turn['response']:
                        self.vocab.add_item(word_token)
                    for list_word in single_turn['constraint_value']:
                        for word in list_word:
                            self.vocab.add_item(word)
            tokenized_data.append(tokenized_dial)
        self._save_tokenized_data(tokenized_data, data_type)
        return tokenized_data

    def _get_encoded_data(self, tokenized_data):
        informable = {
            'weather': ['date', 'location', 'weather_attribute'],
            'navigate': ['poi_type', 'distance'],
            'schedule': ['event', 'date', 'time', 'agenda', 'party', 'room']
        }
        requestable = {
            'weather': ['weather_attribute', 'temperature'],
            'navigate': ['poi', 'traffic', 'address', 'distance'],
            'schedule': ['event', 'date', 'time', 'party', 'agenda', 'room']
        }
        informable_keys = ['date', 'location', 'weather_attribute', 'poi_type', 'distance', 'event', 'date', 'time',
                           'agenda', 'party', 'room']
        constraint_eos = ['EOS_' + k for k in informable_keys]
        encoded_data = []
        for dial in tokenized_data:
            new_dial = []
            for turn in dial:
                turn['constraint'] = self.vocab.sentence_encode(turn['constraint'])
                turn['requested'] = self.vocab.sentence_encode(turn['requested'])
                # turn['bspan'] = turn['constraint'] + turn['requested']
                turn['user'] = self.vocab.sentence_encode(turn['user'])
                turn['response'] = self.vocab.sentence_encode(turn['response'])
                turn['u_len'] = len(turn['user'])
                turn['m_len'] = len(turn['response'])
                turn['degree'] = self._degree_vec_mapping(turn['degree'])
                turn['constraint_key'] = self.vocab.sentence_encode(turn['constraint_key'])
                turn['constraint_value'] = [self.vocab.sentence_encode(c) for c in turn['constraint_value']]
                turn['bspan'] = [w for c in turn['constraint_value'] for w in c] + turn['requested']
                turn['requestable_key'] = self.vocab.sentence_encode(turn['requestable_key'])
                turn['requestable_slot'] = self.vocab.sentence_encode(turn['requestable_slot'])
                turn['requested_7'] = np.asarray(turn['requested_7'])
                turn['response_7'] = np.asarray(turn['response_7'])
                turn['constraint_eos'] = self.vocab.sentence_encode(constraint_eos)
                new_dial.append(turn)
            encoded_data.append(new_dial)
        return encoded_data

    def _get_entity_dict(self, entity_data):
        entity_dict = {}
        for k in entity_data:
            if type(entity_data[k][0]) is str:
                for entity in entity_data[k]:
                    entity = self._lemmatize(self._tokenize(entity))
                    entity_dict[entity] = k
                    if k in ['event', 'poi_type']:
                        entity_dict[entity.split()[0]] = k
                        self.abbr_dict[entity.split()[0]] = entity
            elif type(entity_data[k][0]) is dict:
                for entity_entry in entity_data[k]:
                    for entity_type, entity in entity_entry.items():
                        entity_type = 'poi_type' if entity_type == 'type' else entity_type
                        entity = self._lemmatize(self._tokenize(entity))
                        entity_dict[entity] = entity_type
                        if entity_type in ['event', 'poi_type']:
                            entity_dict[entity.split()[0]] = entity_type
                            self.abbr_dict[entity.split()[0]] = entity
        self.entity_dict = entity_dict

    def Lei_db_degree(self, constraints, ori_constraints, items, intent):
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        distance_list = [
            "1 miles",
            "2 miles",
            "3 miles",
            "4 miles",
            "5 miles",
            "6 miles",
            "7 miles",
            "8 miles",
            "9 miles"
        ]
        ignored_constraints = dict()
        if constraints != ori_constraints:
            diff_constraints = {k: v for k, v in ori_constraints.items() if k not in constraints}
            if intent == 'navigate':
                for k, value in diff_constraints.items():
                    if k == 'distance':
                        if 'est' in value or 'closet' in value:
                            ignored_constraints[k] = 'nearest'
                        elif 'mile' in value:
                            if '1 mile' in value:
                                ignored_constraints[k] = '1 miles'
                            else:
                                for dis in distance_list:
                                    if dis in value:
                                        ignored_constraints[k] = dis
                                        break
                            if k not in ignored_constraints:
                                print(value)
                        else:
                            pass
                    elif k == 'poi_type':
                        if 'chinese' in value:
                            constraints[k] = 'chinese restaurant'
                        elif 'pizza' in value:
                            constraints[k] = 'pizza restaurant'
                        elif 'fast food' in value:
                            constraints[k] = 'fast food'
                        elif 'eat' in value or 'restaurant' in value or 'food' in value or 'lunch' in value:
                            constraints[k] = 'restaurant'  # has bug
                        elif 'park' in value or 'garage' in value or 'fill up' in value:
                            constraints[k] = 'parking garage'
                        elif 'mall' in value:
                            constraints[k] = 'shopping center'
                        elif 'tea' in value or 'coffee' in value or 'cafe' in value or 'barista' in value:
                            constraints[k] = 'coffee or tea place'  # has bug
                        elif 'motel' in value or 'hotel' in value or 'sleep' in value:
                            constraints[k] = 'rest stop'
                        elif 'produce stand' in value:
                            constraints[k] = 'grocery store'
                        elif 'friend' in value:
                            constraints[k] = 'friends house'
                        elif 'house' in value or 'my' in value or 'live' in value:
                            constraints[k] = 'home'
                        else:
                            constraints[k] = value
                            # print(value)
                    else:
                        pass
            elif intent == 'schedule':
                for k, value in diff_constraints.items():
                    if k == 'agenda':
                        pass
                    elif k == 'event':
                        if 'medicine' in value or 'medication' in value or 'pill' in value:
                            constraints[k] = 'taking medicine'
                        elif 'optome' in value or 'eye' in value:
                            constraints[k] = 'optometrist appointment'
                        elif 'dentist' in value or 'dental' in value:
                            constraints[k] = 'dentist appointment'
                        elif 'doctor' in value or 'dr' in value:
                            constraints[k] = 'doctor appointment'
                        elif 'dinner' in value:
                            constraints[k] = 'dinner'
                        elif 'lab' in value or 'blood' in value or 'shot' in value or 'vaccination' in value:
                            constraints[k] = 'lab appointment'
                        elif 'meeting' in value:
                            constraints[k] = 'meeting'
                        elif 'football' in value:
                            constraints[k] = 'football activity'
                        elif 'yoga' in value:
                            constraints[k] = 'yoga activity'
                        elif 'swim' in value:
                            constraints[k] = 'tennis activity'
                        elif 'conference' in value:
                            constraints[k] = 'conference'
                        elif 'tennis' in value or 'tenis' in value:
                            constraints[k] = 'swimming activity'
                        elif 'party' in value or 'parties' in value:
                            constraints[k] = 'party'
                        elif 'appointment' in value:
                            constraints[k] = 'appointment'
                        elif 'workout' in value or 'work out' in value or 'exercise' in value:
                            constraints[k] = 'activity'
                        else:
                            pass
                    elif k == 'time':
                        pass
                    elif k == 'party':
                        if 'mon' in value or 'mom' in value:
                            constraints[k] = 'mother'
                        elif 'father' in value or 'dad' in value:
                            constraints[k] = 'father'
                        elif 'brother' in value:
                            constraints[k] = 'brother'
                        elif 'sister' in value:
                            constraints[k] = 'sister'
                        elif 'boss' in value:
                            constraints[k] = 'boss'
                        elif 'executive' in value:
                            constraints[k] = 'executive team'
                        elif 'vp' in value:
                            constraints[k] = 'vice president'
                        elif 'infrastructure' in value:
                            constraints[k] = 'infrastructure team'
                        else:
                            constraints[k] = 'value'
                    elif k == 'room':
                        if '100' in value:
                            constraints[k] = 'conference room 100'
                        else:
                            pass
                    elif k == 'date':
                        flag = False
                        for day in days:
                            if day in value:
                                constraints[k] = day
                                flag = True
                                break
                            else:
                                pass
                        if flag == False:
                            for i in range(32, 0, -1):
                                str_t = str(i)
                                if i == 1 or i == 21:
                                    date = str_t + 'st'
                                elif i == 2 or i == 22:
                                    date = str_t + 'nd'
                                elif i == 3 or i == 23:
                                    date = str_t + 'rd'
                                else:
                                    date = str_t + 'th'
                                if date in value:
                                    constraints[k] = date
                                    flag = True
                                    break
                        if flag == False:
                            for i in range(32, 0, -1):
                                str_t = str(i)
                                if str_t in value:
                                    if i == 1 or i == 21:
                                        date = str_t + 'st'
                                    elif i == 2 or i == 22:
                                        date = str_t + 'nd'
                                    elif i == 3 or i == 23:
                                        date = str_t + 'rd'
                                    else:
                                        date = str_t + 'th'
                                    constraints[k] = date
                                    flag = True
                                    break
                                else:
                                    pass
                        else:
                            pass
                        if flag == False:
                            if 'tenth' in value:
                                constraints[k] = '10th'
                            else:
                                # print(value)
                                pass
                        else:
                            pass
                            # print(constraints[k], value)
                    else:
                        pass
            else:  # weather
                for k, value in diff_constraints.items():
                    if k == 'date':
                        if value in ['during the week', 'entire week', 'any day', 'any day this week', 'weekly',
                                     'whole week', '7 days', 'all week', 'nest week', '7 day',
                                     'upcoming week', 'weekly (this week)', 'next 7 days', 'the whole week', 'anytime',
                                     'the week', 'this week', 'seven days',
                                     'seven day', 'next 7 day period', 'next seven days', 'this upcoming week',
                                     '7 day forecast', 'full weekly', 'days']:
                            constraints[k] = 'week'
                        elif value in ['48 hours', 'next 48 hours', '2 days', 'next two days', '48 hour', '2 day',
                                       'next 2 days']:
                            constraints[k] = 'two day'
                        elif value in ['this weekend', 'next weekend']:
                            constraints[k] = 'weekend'
                        elif value in ['next several days', 'several days']:
                            constraints[k] = 'next few days'
                        elif value in ['right now', 'currently', 'now', 'current']:
                            constraints[k] = 'today'
                        elif value in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                            constraints[k] = value
                        else:
                            print(value)
                    elif k == 'weather_attribute':
                        if value in ['high temperature', 'above 86 degrees', '90f', 'high', 'high temperatures']:
                            constraints[k] = 'highest temperature'
                        elif value in ['low temperature']:
                            constraints[k] = 'lowest temperature'
                        elif value in ['climate', '7 day forecast', 'forecast', 'temperature', 'weekly weather report']:
                            constraints[k] = 'temperature'
                        elif value in ['snowing', 'snowy']:
                            constraints[k] = 'snow'
                        elif value in ['raining', 'chance of rain']:
                            constraints[k] = 'rain'
                        elif value == 'how warm':
                            constraints[k] = 'warm'
                        elif value == 'hailing':
                            constraints[k] = 'hail'
                        elif value == 'clear':
                            constraints[k] = "clear skies"
                        else:
                            pass
                            # print(value)
                    else:  # k==location
                        if value == 'new york city':
                            constraints[k] = 'new york'
                        elif value == 'redwood':
                            constraints[k] = 'redwood city'
                        elif value == 'chicago':
                            constraints[k] = 'downtown chicago'
                        elif value in ['outside', 'local', 'my city', 'city', 'the city']:
                            pass
                        else:
                            pass
                            #print(value)

        cnt = 0
        if len(constraints) == 0 or constraints == None:
            if items != None:
                return len(items)
            else:
                return 0
        cand_items = []
        if items != None:
            for item in items:
                if item == None:
                    continue
                flg = True
                for k, c in constraints.items():
                    if k not in item:
                        ignored_constraints[k] = c
                    else:
                        if intent == 'navigate' and c == 'restaurant':
                            if c not in item[k]:
                                flg = False
                                break
                        elif intent == 'schedule' and c in ['appointment', 'activity']:
                            if c not in item[k]:
                                flg = False
                                break
                        else:
                            if c != item[k]:
                                flg = False
                                break
                if flg:
                    cnt += 1
                    cand_items.append(item)

        if len(ignored_constraints) > 0:
            if intent == 'navigate':
                if cnt == 0:
                    return 0
                elif cnt == 1:
                    if 'distance' in ignored_constraints:
                        distance_constraint = ignored_constraints['distance']
                        if distance_constraint != 'nearest':
                            target_index = distance_list.index(distance_constraint)
                            target_distance = distance_list[:target_index + 1]
                            distance_cnt = 0
                            for r in cand_items:
                                if r['distance'] in target_distance:
                                    distance_cnt += 1
                            return distance_cnt
                        else:
                            return cnt
                    else:
                        return cnt
                else:
                    if 'distance' in ignored_constraints:
                        distance_constraint = ignored_constraints['distance']
                        if distance_constraint == 'nearest':
                            distance_values = [distance_list.index(r['distance']) for r in cand_items]
                            min_value = min(distance_values)
                            distance_cnt = 0
                            for v in distance_values:
                                if v == min_value:
                                    distance_cnt += 1
                            return distance_cnt

                        else:
                            target_index = distance_list.index(distance_constraint)
                            target_distance = distance_list[:target_index + 1]
                            distance_cnt = 0
                            for r in cand_items:
                                if r['distance'] in target_distance:
                                    distance_cnt += 1
                            return distance_cnt
                    else:
                        return cnt

            elif intent == 'weather':
                weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
                weekend = ['saturday', 'sunday']
                if cnt == 0:
                    return 0
                elif cnt == 1:
                    cand_item = cand_items[0]
                    if 'date' in ignored_constraints:
                        if ignored_constraints['date'] in cand_item:
                            date_value = cand_item[
                                ignored_constraints['date']]  # date_value maybe weather or today's value
                            if len(ignored_constraints) == 1:  # only date constraint
                                return cnt
                            else:
                                weather_constraint = ignored_constraints['weather_attribute']
                                if weather_constraint in ["lowest temperature", "highest temperature"]:
                                    return cnt
                                else:
                                    if ignored_constraints['date'] == 'today':
                                        date_value = cand_item[cand_item['today']]
                                    # print(date_value)
                                    if weather_constraint in date_value:
                                        return 1
                                    else:
                                        return 0
                        else:  # 'date' value is not in item
                            if ignored_constraints['date'] in ['rest', 'next week', 'week', 'next few day', 'two day',
                                                               'tomorrow', 'weekend']:
                                if len(ignored_constraints) == 1:  # only date constraint
                                    if ignored_constraints['date'] == 'tomorrow':
                                        return 1
                                    elif ignored_constraints['date'] in ['two day', 'weekend']:
                                        return 2
                                    else:
                                        return 5
                                else:
                                    weather_constraint = ignored_constraints['weather_attribute']
                                    if weather_constraint in ["lowest temperature", "highest temperature"]:
                                        return 1
                                    else:
                                        if ignored_constraints['date'] == 'tomorrow':
                                            target_days = [days[days.index(cand_item['today']) + 1]]
                                        elif ignored_constraints['date'] == 'two day':
                                            target_days = days[
                                                          days.index(cand_item['today']):days.index(
                                                              cand_item['today']) + 2]
                                        elif ignored_constraints['date'] == 'weekend':
                                            target_days = weekend
                                        elif ignored_constraints['date'] in ['week', 'next week']:
                                            target_days = days
                                        else:
                                            target_days = days[
                                                          days.index(cand_item['today']) + 1:]
                                        weather_cnt = 0
                                        for date, weather in cand_item.items():
                                            if date in target_days and weather_constraint in weather:
                                                weather_cnt += 1
                                        return weather_cnt
                            else:
                                print('ERROR kvret data db search:: date value is not correct: ',
                                      ignored_constraints['date'])
                    else:  # only weather in constraints
                        weather_constraint = ignored_constraints['weather_attribute']
                        if weather_constraint in ["lowest temperature", "highest temperature"]:
                            return cnt
                        else:
                            weather_cnt = 0
                            for date, weather in cand_item.items():
                                if weather_constraint in weather:
                                    weather_cnt += 1
                            # print('multiple date ', weather_cnt)
                            return weather_cnt
                else:  # cnt > 1 multiple location
                    # print('ignore weather multiple location')
                    return cnt
            else:
                print(intent)

        return cnt

    def db_degree(self, constraints, items):
        cnt = 0
        if len(constraints) == 0 or constraints == None:
            if items != None:
                return len(items)
            else:
                return 0
        if items != None:
            for item in items:
                if item == None:
                    continue
                flg = True
                for c in constraints:
                    if c not in item:
                        flg = False
                        break
                if flg:
                    cnt += 1
        return cnt


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)
    if maxlen is not None and cfg.truncated:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def get_glove_matrix(vocab, initial_embedding_np):
    """
    return a glove embedding matrix
    :param self:
    :param glove_file:
    :param initial_embedding_np:
    :return: np array of [V,E]
    """
    if os.path.exists(cfg.vocab_emb):
        vec_array = np.load(cfg.vocab_emb)
        old_avg = np.average(vec_array)
        old_std = np.std(vec_array)
        logging.info('embedding.  mean: %f  std %f' % (old_avg, old_std))
        return vec_array
    else:
        ef = open(cfg.glove_path, 'r')
        cnt = 0
        vec_array = initial_embedding_np
        old_avg = np.average(vec_array)
        old_std = np.std(vec_array)
        vec_array = vec_array.astype(np.float32)
        new_avg, new_std = 0, 0

        for line in ef.readlines():
            line = line.strip().split(' ')
            word, vec = line[0], line[1:]
            vec = np.array(vec, np.float32)
            word_idx = vocab.encode(word)
            if word.lower() in ['unk', '<unk>'] or word_idx != vocab.encode('<unk>'):
                cnt += 1
                vec_array[word_idx] = vec
                new_avg += np.average(vec)
                new_std += np.std(vec)
        new_avg /= cnt
        new_std /= cnt
        ef.close()
        logging.info('%d known embedding. old mean: %f new mean %f, old std %f new std %f' % (
            cnt, old_avg, new_avg, old_std, new_std))
        np.save(cfg.vocab_emb, vec_array)
        return vec_array
