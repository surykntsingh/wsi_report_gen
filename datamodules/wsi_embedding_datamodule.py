import torch
from torch.utils.data import Subset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from embedding_datasets.embedding_dataset import EmbeddingDataset, EmbeddingPredictDataset


class PatchEmbeddingDataModule(pl.LightningDataModule):

    def __init__(self,args, tokenizer, split_frac, shuffle = False):
        super().__init__()
        self.test_ds = None
        self.val_ds = None
        self.train_ds = None
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.embeddings_path = args.embeddings_path
        self.reports_json_path = args.reports_json_path
        self.max_seq_length = args.max_seq_length
        self.split_frac = split_frac
        self.tokenizer = tokenizer
        self.embeddings_path_2 = args.embeddings_path_2
        self.gecko_emb_path = args.gecko_emb_path

    def setup(self, stage=None):
        dataset = EmbeddingDataset(self.embeddings_path, self.reports_json_path, self.tokenizer,
                              self.max_seq_length, self.embeddings_path_2, self.gecko_emb_path)
        # print(dataset[0][1])
        self.train_ds, self.val_ds, self.test_ds = random_split(dataset, self.split_frac)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn = self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn = self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn = self.collate_fn)

    @staticmethod
    def collate_fn(batch, device='cuda'):
        slide_ids, feats_1, feats_2, emb_g, emb_gc, coord_feats, report_ids, report_masks, seq_length = zip(*batch)
        feats1_pad = pad_sequence(feats_1, batch_first=True).to(device)
        feats2_pad = pad_sequence(feats_2, batch_first=True).to(device)
        emb_g_pad = pad_sequence(emb_g, batch_first=True).to(device)
        emb_gc_pad = pad_sequence(emb_gc, batch_first=True).to(device)
        report_ids = torch.LongTensor(report_ids).to(device)
        # dummy_feat = torch.randn(patch_feats_pad.shape)
        # coord_feats_pad =  pad_sequence(coord_feats, batch_first=True)
        # patch_mask = torch.zeros(patch_feats_pad.shape[:2], dtype=torch.float32)
        # for i, p in enumerate(patch_feats):
        #     patch_mask[i, :p.shape[0]] = 1
        # print(slide_ids, len(patch_feats1), len(patch_feats2))
        return (slide_ids, feats1_pad, feats2_pad, emb_g_pad, emb_gc_pad, report_ids,
                torch.FloatTensor(report_masks), seq_length)


class EmbeddingDataModule(PatchEmbeddingDataModule):

    def __init__(self, args, tokenizer, split_frac, train_idx, test_idx, shuffle=False):
        super().__init__(args, tokenizer, split_frac, shuffle)
        self.train_idx = train_idx
        self.test_idx = test_idx

        print(f'self.train_idx: {self.train_idx}')
        print(f'self.test_idx: {self.test_idx}')

    def setup(self, stage=None):
        dataset = EmbeddingDataset(self.embeddings_path, self.reports_json_path, self.tokenizer,
                                   self.max_seq_length, self.embeddings_path_2, self.gecko_emb_path)
        self.train_ds, self.val_ds = random_split(Subset(dataset, self.train_idx), self.split_frac)
        self.test_ds = Subset(dataset, self.test_idx)


class PatchEmbeddingDataPredictModule(pl.LightningDataModule):

    def __init__(self,args, tokenizer, shuffle = False, slide_ids=None):
        super().__init__()
        self.predict_ds = None
        self.__batch_size = args.batch_size
        self.__shuffle = shuffle
        self.__num_workers = args.num_workers
        self.__embeddings_path = args.predict_embeddings_path_1
        self.__max_seq_length = args.max_seq_length
        self.__tokenizer = tokenizer
        self.__embeddings_path_2 = args.predict_embeddings_path_2
        self.__gecko_emb_predict_path = args.gecko_predict_emb_path
        self.__slide_ids = slide_ids

    def setup(self, stage=None):
        self.predict_ds = EmbeddingPredictDataset(self.__embeddings_path, self.__tokenizer,
                              self.__max_seq_length, self.__embeddings_path_2, self.__gecko_emb_predict_path, self.__slide_ids)
        print(f'predict ds: {len(self.predict_ds)}')


    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.__batch_size, shuffle=self.__shuffle, collate_fn = self.collate_fn)


    @staticmethod
    def collate_fn(batch, device='cuda'):
        slide_ids, feats_1, feats_2,emb_g,emb_gc = zip(*batch)
        feats1_pad = pad_sequence(feats_1, batch_first=True).to(device)
        feats2_pad = pad_sequence(feats_2, batch_first=True).to(device)
        emb_g_pad = pad_sequence(emb_g, batch_first=True).to(device)
        emb_gc_pad = pad_sequence(emb_gc, batch_first=True).to(device)

        return slide_ids, feats1_pad, feats2_pad, emb_g_pad, emb_gc_pad