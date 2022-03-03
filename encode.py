import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import types
import numpy as np
from model import base_model, embedding_layer, lstm_layer
from word_tokenizer import WordTokenizer
from transformers import BertTokenizer,BertModel
class base_encoder(base_model):

    def __init__(self,
                 token2id=None,
                 word2vec=None,
                 word_size=50,
                 max_length=128,
                 blank_padding=True):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
        """
        # hyperparameters
        super(base_encoder, self).__init__()

        if isinstance(token2id, list):
            self.token2id = {}
            for index, token in enumerate(token2id):
                self.token2id[token] = index
        else:
            self.token2id = token2id

        self.max_length = max_length
        self.num_token = len(self.token2id)

        if isinstance(word2vec, type(None)):
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]

        self.blank_padding = blank_padding

        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1

        if not isinstance(word2vec, type(None)):
            word2vec = torch.from_numpy(word2vec)
            if self.num_token == len(word2vec) + 2:
                unk = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                blk = torch.zeros(1, self.word_size)
                self.word2vec = (torch.cat([word2vec, unk, blk], 0)).numpy()
            else:
                self.word2vec = word2vec
        else:
            self.word2vec = None

        self.tokenizer = WordTokenizer(vocab=self.token2id, unk_token="[UNK]")

    def set_embedding_layer(self, embedding_layer):
        self.embedding_layer = embedding_layer

    def set_encoder_layer(self, encoder_layer):
        self.encoder_layer = encoder_layer

    def forward(self, token, pos1, pos2):
        pass

    def tokenize(self, sentence):
        """
        Args:
            item: input instance, including sentence, entity positions, etc.
            is_token: if is_token == True, sentence becomes an array of token
        Return:
            index number of tokens and positions
        """
        tokens = self.tokenizer.tokenize(sentence)
        length = min(len(tokens), self.max_length)
        # Token -> index

        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'],
                                                                  self.token2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id=self.token2id['[UNK]'])

        if (len(indexed_tokens) > self.max_length):
            indexed_tokens = indexed_tokens[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
        length = torch.tensor([length]).long()
        return indexed_tokens, length

class lstm_encoder(base_encoder):

    def __init__(self, token2id = None, word2vec = None, word_size = 50, max_length = 128,
            pos_size = None, hidden_size = 230, dropout = 0, bidirectional = True, num_layers = 1, config = None):
        super(lstm_encoder, self).__init__(token2id, word2vec, word_size, max_length, blank_padding = False)
        self.config = config
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.word_size = word_size
        self.pos_size = pos_size
        self.input_size = word_size
        if bidirectional:
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size
        if pos_size != None:
            self.input_size += 2 * pos_size
        self.embedding_layer = embedding_layer(self.word2vec, max_length, word_size, None, False)
        self.encoder_layer = lstm_layer(max_length, self.input_size, hidden_size, dropout, bidirectional, num_layers, config)

    def forward(self, inputs, lengths = None):
        inputs, lengths, inputs_indexs = self.encoder_layer.pad_sequence(inputs, padding_value = self.token2id['[PAD]'])
        inputs = inputs.to(self.config['device'])
        x = self.embedding_layer(inputs)
        x = self.encoder_layer(x, lengths, inputs_indexs)
        return x

class BERTSentenceEncoder(nn.Module):
    def __init__(self, config,ckptpath=None):
        nn.Module.__init__(self)
        if ckptpath != None:
            ckpt = torch.load(ckptpath)
            self.bert = BertModel.from_pretrained(config["pretrained_model"],state_dict=ckpt["bert-base"])
        else:
            self.bert = BertModel.from_pretrained(config["pretrained_model"])
        print("aaaaaaaaaaaaaaaaaaaa")
        unfreeze_layers = ['layer.11', 'bert.pooler.', 'out.']
        print(unfreeze_layers)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        print("freeze finished")
        #'''
        ###5 no freeze
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrained_model"])
        self.output_size = 768
    def forward(self, inputs, mask):
        outputs = self.bert(inputs, attention_mask=mask)
        #print("outputs[0].shape: ",outputs[0].shape)
        #print("outputs[1].shape: ",outputs[1].shape)
        return outputs[1]

class Transformer_Encoder(base_encoder):
    #def __init__(self, num_layers, d_model, vocab_size, h, dropout):
    def __init__(self, token2id=None, word2vec=None, word_size=300, max_length=128, dropout=0, head = 4, num_layers=1, config=None):
        super(Transformer_Encoder, self).__init__(token2id, word2vec, word_size, max_length, blank_padding=False)

        self.config = config
        self.max_length = max_length
        self.hidden_size = word_size  ####d_model
        self.output_size = word_size
        self.embedding_layer = EmbeddingLayer(self.word2vec, max_length, word_size, False)

        self.layers = nn.ModuleList([EncoderLayer(word_size, head, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(word_size)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()


    def forward(self, x, mask):
        #print("---------------------")
        #print(x.shape)
        #print(mask.shape)
        batch_size = x.size(0)
        max_enc_len = x.size(1)

        assert max_enc_len == self.max_length

        pos_idx = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(self.config['device'])

        y = self.embedding_layer(x, pos_idx[:, :max_enc_len])
        #print(y.shape)
        assert y.size(1) == mask.size(-1)
        mask = (mask[:, :max_enc_len] == 0)
        mask = mask.view(batch_size, 1, 1, max_enc_len)
        for layer in self.layers:
            y = layer(y, mask)

        encoder_outputs = self.norm(y)
        #print(encoder_outputs.shape)
        sequence_output = encoder_outputs[:,0]
        pooled_output = self.activation(self.dense(sequence_output))
        return pooled_output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.pw_ffn)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.head_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x, l in zip((query, key, value), self.head_projs)]

        attn_feature, _ = scaled_attention(query, key, value, mask)

        attn_concated = attn_feature.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.fc(attn_concated)


def scaled_attention(query, key, value, mask):
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    scores.masked_fill_(mask, float('-inf'))
    attn_weight = F.softmax(scores, -1)
    attn_feature = attn_weight.matmul(value)

    return attn_feature, attn_weight


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.mlp(x)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)

    def incremental_forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)


def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

class EmbeddingLayer(nn.Module):
    #def __init__(self, n_words, d_model, max_length, pad_idx, learned_pos_embed, load_pretrained_embed):
    def __init__(self, word_vec_mat, max_length, word_embedding_dim=300, requires_grad=True):
        super(EmbeddingLayer, self).__init__()

        self.max_length = max_length

        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = word_embedding_dim
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.token_embed = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim,
                                           padding_idx=word_vec_mat.shape[0] - 1)
        self.token_embed.weight.data.copy_(word_vec_mat)
        self.token_embed.weight.requires_grad = requires_grad

        self.pos_embed = nn.Embedding(max_length, self.pos_embedding_dim, padding_idx=max_length - 1)

    def forward(self, x, pos):
        if len(x.size()) == 2:
            y = self.token_embed(x) + self.pos_embed(pos)
        return y

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps)
    return m