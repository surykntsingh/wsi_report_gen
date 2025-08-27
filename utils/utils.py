import copy
import json
import re
import yaml
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


def read_json_file(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return d

def write_json_file(obj, out_path):
    with open(out_path, 'w') as f:
        json.dump(obj, f, indent=4)

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    if length < alpha:
        penalty = -1000
    else:
        penalty = logprobs / length
    return penalty


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def pad_tokens(att_feats):
    # ---->pad
    H = att_feats.shape[1]
    _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    add_length = _H * _W - H
    att_feats = torch.cat([att_feats, att_feats[:, :add_length, :]], dim=1)  # [B, N, L]
    return att_feats


def sort_pack_padded_sequence(input, lengths):
    lengths = lengths.cpu()
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    # print(module, att_feats, att_masks)
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


def save_model(args, trainer):
    print(f'Saving model at path: {args.model_save_path}')
    trainer.save_model(args.model_save_path)


def extract_fields(report_text):
    fields = {}
    # Normalize whitespace
    text = " ".join(report_text.strip().split())
    # ---- Biopsy site and mode (header) ----
    m = re.match(r'^\s*([^,]+),\s*([^;]+);', text, re.IGNORECASE)
    if m:
        fields['Biopsy site'] = m.group(1).strip()
        fields['Biopsy mode'] = m.group(2).strip()
        text = text[m.end():].strip()
    # ---- Diagnosis ----
    diag_m = re.match(r'^(.*?)(?:\(|$)', text)
    if diag_m:
        diag = diag_m.group(1).strip().rstrip(',;')
        # Remove a trailing ", grade X" if present
        diag = re.sub(r',\s*grade\s+[IVX]+', '', diag, flags=re.IGNORECASE).strip()
        if diag:
            fields['Diagnosis'] = diag
    # ---- Optional Fields ----
    # Gleason score (e.g., "Gleason 3+4")
    gleason = re.search(r'gleason\s*(?:score)?\s*[:\s]*([\d+]+)', text, re.IGNORECASE)
    if gleason:
        fields['Gleason score'] = gleason.group(1)
    # Tumor volume (number or % after "tumor volume")
    vol = re.search(r'\btumor\s+volume\s*[:\s]*([\d\.]+\s*%?)', text, re.IGNORECASE)
    if vol:
        fields['Tumor volume'] = vol.group(1).strip()
    # Tumor grade (Roman numeral)
    grade = re.search(r'grade\s+([IVX]+)\b', text, re.IGNORECASE)
    if grade:
        fields['Tumor grade'] = grade.group(1)

    nuclear_grade = re.search(r'\bnuclear grade\s+([IVX]+)\b', text, re.IGNORECASE)
    if nuclear_grade:
        fields['Nuclear grade'] = nuclear_grade.group(1)

    tubule = re.search(r'\btubule formation\s+([IVX]+)\b', text, re.IGNORECASE)
    if nuclear_grade:
        fields['Tubule formation'] = tubule.group(1)

    mitosis = re.search(r'\bmitosis\s+([IVX]+)\b', text, re.IGNORECASE)
    if nuclear_grade:
        fields['Mitosis'] = mitosis.group(1)

    # Margins: find any phrase containing "margin"
    if re.search(r'\bmargin', text, re.IGNORECASE):
        margins = re.findall(r'\bmargin[s]?\b[^,;.]*', text, re.IGNORECASE)
        if margins:
            fields['Margins'] = "; ".join([m.strip() for m in margins])
    # Histologic subtype: e.g. after 'of'
    if 'Diagnosis' in fields:
        subtype = re.search(r'of\s+(.+)', fields['Diagnosis'], re.IGNORECASE)
        if subtype:
            sub = subtype.group(1).strip()
            fields['Histologic subtype'] = sub
    # Lymphovascular invasion (LVI): present or absent
    if re.search(r'\blymphovascular\b|\bLVI\b', text, re.IGNORECASE):
        if re.search(r'\bno\b\s+evidence\s+of\s+lymphovascular', text, re.IGNORECASE):
            fields['Lymphovascular invasion'] = 'Absent'
        else:
            fields['Lymphovascular invasion'] = 'Present'
    # Perineural invasion (PNI)
    if re.search(r'\bperineural\b|\bPNI\b', text, re.IGNORECASE):
        if re.search(r'\bno\b\s+evidence\s+of\s+perineural', text, re.IGNORECASE):
            fields['Perineural invasion'] = 'Absent'
        else:
            fields['Perineural invasion'] = 'Present'
    # DCIS presence
    if re.search(r'\bDCIS\b', text, re.IGNORECASE):
        fields['DCIS presence'] = 'Present'
    return fields


def get_params_for_key(file_path: str, param_key: str):
    """utility function to get the params
    from params.yaml file given the key

    :param file_path: path of the params file
    :param param_key: key to read
    :returns: params dict

    """

    with open(file_path) as f:
        all_params = yaml.safe_load(f)
        params = all_params[param_key]

        print(f'params: {params}')
        return SimpleNamespace(**params)

def copy_yaml(source_path, target_dir, file_name='config.yaml'):
    target_path = f'{target_dir}/{file_name}'
    with open(source_path) as source_file:
        params = yaml.safe_load(source_file)

    with open(target_path, 'w') as destination_file:
        yaml.dump(params, destination_file, default_flow_style=False, sort_keys=False)