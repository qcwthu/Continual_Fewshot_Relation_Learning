import torch
import torch.nn as nn
import os
import json
import numpy as np

class base_model(nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def get_parameters(self, mode = "numpy", param_dict = None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()


class embedding_layer(base_model):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=None, requires_grad=True):
        super(embedding_layer, self).__init__()
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim

        # Word embedding
        # unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        # blk = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim,
                                           padding_idx=word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data.copy_(word_vec_mat)
        self.word_embedding.weight.requires_grad = requires_grad
        # Position Embedding
        if self.pos_embedding_dim != None:
            self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
            self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

    def forward(self, word, pos1=None, pos2=None):
        if pos1 != None and pos2 != None and self.pos_embedding_dim != None:
            x = torch.cat([self.word_embedding(word),
                           self.pos1_embedding(pos1),
                           self.pos2_embedding(pos2)], 2)
        else:
            x = self.word_embedding(word)
        return x


class lstm_layer(base_model):

    def __init__(self, max_length=128, input_size=50, hidden_size=256, dropout=0, bidirectional=True, num_layers=1,
                 config=None):
        """
        Args:
            input_size: dimention of input embedding
            hidden_size: hidden size
            dropout: dropout layer on the outputs of each RNN layer except the last layer
            bidirectional: if it is a bidirectional RNN
            num_layers: number of recurrent layers
            activation_function: the activation function of RNN, tanh/relu
        """
        super(lstm_layer, self).__init__()
        self.device = config['device']
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.input_size = input_size
        if bidirectional:
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers,
                            dropout=dropout)

    def init_hidden(self, batch_size=1, device='cpu'):
        self.hidden = (torch.zeros(2, batch_size, self.hidden_size).to(device),
                       torch.zeros(2, batch_size, self.hidden_size).to(device))

    def forward(self, inputs, lengths, inputs_indexs):
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths)
        lstm_out, hidden = self.lstm(packed_embeds, self.hidden)
        permuted_hidden = hidden[0].permute([1, 0, 2]).contiguous()
        permuted_hidden = permuted_hidden.view(-1, self.hidden_size * 2)
        output_embedding = permuted_hidden[inputs_indexs]
        return output_embedding

    def ranking_sequence(self, sequence):
        word_lengths = torch.tensor([len(sentence) for sentence in sequence])
        rankedi_word, indexs = word_lengths.sort(descending=True)
        ranked_indexs, inverse_indexs = indexs.sort()
        sequence = [sequence[i] for i in indexs]
        return sequence, inverse_indexs

    def pad_sequence(self, inputs, padding_value=0):
        self.init_hidden(len(inputs), self.device)
        inputs, inputs_indexs = self.ranking_sequence(inputs)
        lengths = [len(data) for data in inputs]
        pad_inputs = torch.nn.utils.rnn.pad_sequence(inputs, padding_value=padding_value)
        return pad_inputs, lengths, inputs_indexs


class proto_softmax_layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __distance__(self, rep, rel):
        '''
        rep_ = rep.view(rep.shape[0], 1, rep.shape[-1])
        rel_ = rel.view(1, -1, rel.shape[-1])
        dis = (rep_ * rel_).sum(-1)
        return dis
        '''
        rep_norm = rep / rep.norm(dim=1)[:, None]
        rel_norm = rel / rel.norm(dim=1)[:, None]
        res = torch.mm(rep_norm, rel_norm.transpose(0, 1))
        return res

    def __init__(self, sentence_encoder, num_class, id2rel, drop=0, config=None, rate=1.0):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super(proto_softmax_layer, self).__init__()

        self.config = config
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.hidden_size = self.sentence_encoder.output_size
        self.fc = nn.Linear(self.hidden_size, self.num_class, bias=False)
        self.drop = nn.Dropout(drop)
        self.id2rel = id2rel
        self.rel2id = {}
        for id, rel in id2rel.items():
            self.rel2id[rel] = id

    def set_memorized_prototypes(self, protos):
        self.prototypes = protos.detach().to(self.config['device'])

    def get_feature(self, sentences, length=None):
        rep = self.sentence_encoder(sentences, length)
        return rep.cpu().data.numpy()

    def get_mem_feature(self, rep):
        dis = self.mem_forward(rep)
        return dis.cpu().data.numpy()

    def forward(self, sentences, length=None):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(sentences, length)  # (B, H)
        repd = self.drop(rep)
        logits = self.fc(repd)
        return logits, rep

    def mem_forward(self, rep):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        dis_mem = self.__distance__(rep, self.prototypes)
        return dis_mem


class proto_softmax_layer_bert(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __distance__(self, rep, rel):
        rep_norm = rep / rep.norm(dim=1)[:, None]
        rel_norm = rel / rel.norm(dim=1)[:, None]
        res = torch.mm(rep_norm, rel_norm.transpose(0, 1))
        return res

    def __init__(self, sentence_encoder, num_class, id2rel, drop=0, config=None, rate=1.0):
        super(proto_softmax_layer_bert, self).__init__()

        self.config = config
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.hidden_size = self.sentence_encoder.output_size
        self.fc = nn.Linear(self.hidden_size, self.num_class, bias=False)
        self.drop = nn.Dropout(drop)
        self.id2rel = id2rel
        self.rel2id = {}
        for id, rel in id2rel.items():
            self.rel2id[rel] = id

    def set_memorized_prototypes(self, protos):
        self.prototypes = protos.detach().to(self.config['device'])

    def get_feature(self, sentences, mask):
        rep = self.sentence_encoder(sentences, mask)
        return rep.cpu().data.numpy()

    def get_mem_feature(self, rep):
        dis = self.mem_forward(rep)
        return dis.cpu().data.numpy()

    def forward(self, sentences, mask):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(sentences, mask)  # (B, H)
        repd = self.drop(rep)
        logits = self.fc(repd)
        return logits, rep

    def mem_forward(self, rep):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        dis_mem = self.__distance__(rep, self.prototypes)
        return dis_mem


class proto_softmax_layer_transformer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __distance__(self, rep, rel):
        rep_norm = rep / rep.norm(dim=1)[:, None]
        rel_norm = rel / rel.norm(dim=1)[:, None]
        res = torch.mm(rep_norm, rel_norm.transpose(0, 1))
        return res

    def __init__(self, sentence_encoder, num_class, id2rel, drop=0, config=None, rate=1.0):
        super(proto_softmax_layer_transformer, self).__init__()

        self.config = config
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.hidden_size = self.sentence_encoder.output_size
        self.fc = nn.Linear(self.hidden_size, self.num_class, bias=False)
        self.drop = nn.Dropout(drop)
        self.id2rel = id2rel
        self.rel2id = {}
        for id, rel in id2rel.items():
            self.rel2id[rel] = id

    def set_memorized_prototypes(self, protos):
        self.prototypes = protos.detach().to(self.config['device'])

    def get_feature(self, sentences, mask):
        rep = self.sentence_encoder(sentences, mask)
        return rep.cpu().data.numpy()

    def get_mem_feature(self, rep):
        dis = self.mem_forward(rep)
        return dis.cpu().data.numpy()

    def forward(self, sentences, mask):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(sentences, mask)  # (B, H)
        repd = self.drop(rep)
        logits = self.fc(repd)
        return logits, rep

    def mem_forward(self, rep):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        dis_mem = self.__distance__(rep, self.prototypes)
        return dis_mem