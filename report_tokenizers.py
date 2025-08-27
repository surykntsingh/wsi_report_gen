import os
import json
import re
from collections import Counter

from utils.utils import read_json_file


class Tokenizer:

    def __init__(self, reports_json_path, threshold=1):
        self.__threshold = threshold
        self.__pattern = re.compile(r'\s+|[\w]+|[^\w\s]', re.UNICODE)
        self.__token2idx, self.__idx2token = self.create_vocabulary(reports_json_path)

    def create_vocabulary(self, reports_json_path):
        total_tokens = []
        reports = read_json_file(reports_json_path)

        for report in reports:
            tokens = self.__split_text(report['report'])
            # for token in tokens:
            #     total_tokens.append(token)
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.__threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token

        return token2idx, idx2token

    def get_token_by_id(self, id):
        return self.__idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.__token2idx:
            return self.__token2idx['<unk>']
        return self.__token2idx[token]

    def get_vocab_size(self):
        return len(self.__token2idx)

    def __split_text(self, text):
        # text = text.lower()
        # return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

        return [m.group(0) for m in self.__pattern.finditer(text)]

    def __call__(self, report):
        tokens = self.__split_text(report)
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        # print(ids)
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                # if i >= 1:
                #     txt += ' '
                txt += self.get_token_by_id(idx)
            else:
                break
        return txt

    def batch_decode(self, ids_batch):
        # print(f'ids_batch: {ids_batch}, {ids_batch.shape}')
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out