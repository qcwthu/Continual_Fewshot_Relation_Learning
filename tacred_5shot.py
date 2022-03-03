import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import json
import gc
from tqdm import tqdm
from sklearn.cluster import KMeans
from encode import lstm_encoder
from dataprocess_tacred import data_sampler
from model import proto_softmax_layer
from dataprocess_tacred import get_data_loader
from transformers import BertTokenizer,BertModel
from util import set_seed,process_data,getnegfrombatch,select_similar_data_new_tac
import faiss


def eval_model(config, basemodel, test_set, mem_relations):
    print("One eval")
    print("test data num is:\t",len(test_set))
    basemodel.eval()

    test_dataloader = get_data_loader(config, test_set, shuffle=False, batch_size=30)
    allnum= 0.0
    correctnum = 0.0
    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
               typelabels) in enumerate(test_dataloader):

        logits, rep = basemodel(sentences, lengths)

        distances = basemodel.get_mem_feature(rep)
        short_logits = distances

        #short_logits = logits
        for index, logit in enumerate(logits):
            score = short_logits[index]  # logits[index] + short_logits[index] + long_logits[index]
            allnum += 1.0
            golden_score = score[labels[index]]
            max_neg_score = -2147483647.0
            for i in neg_labels[index]:  # range(num_class):
                if (i != labels[index]) and (score[i] > max_neg_score):
                    max_neg_score = score[i]
            if golden_score > max_neg_score:
                correctnum += 1
    acc = correctnum / allnum
    print(acc)
    basemodel.train()
    return acc

def get_memory(config, model, proto_set):
    memset = []
    resset = []
    rangeset= [0]
    for i in proto_set:
        #print(i)
        memset += i
        rangeset.append(rangeset[-1] + len(i))
    data_loader = get_data_loader(config, memset, False, False)
    features = []
    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
               typelabels) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
    features = np.concatenate(features)

    protos = []
    #print ("proto_instaces:%d"%len(features))
    for i in range(len(proto_set)):
        protos.append(torch.tensor(features[rangeset[i]:rangeset[i+1],:].mean(0, keepdims = True)))
    protos = torch.cat(protos, 0)
    #print(protos.shape)
    return protos

def select_data(mem_set, proto_memory, config, model, divide_train_set, num_sel_data, current_relations, selecttype):
    ####select data according to selecttype
    #selecttype is 0: cluster for every rel
    #selecttype is 1: use ave embedding
    rela_num = len(current_relations)
    for i in range(0, rela_num):
        thisrel = current_relations[i]
        if thisrel in mem_set.keys():
            #print("have set mem before")
            mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}}
            proto_memory[thisrel].pop()
        else:
            mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}}
        thisdataset = divide_train_set[thisrel]
        data_loader = get_data_loader(config, thisdataset, False, False)
        features = []
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext,  lengths,
                   typelabels) in enumerate(data_loader):
            feature = model.get_feature(sentences, lengths)
            features.append(feature)
        features = np.concatenate(features)
        #print(features.shape)
        num_clusters = min(num_sel_data, len(thisdataset))
        if selecttype == 0:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            distances = kmeans.fit_transform(features)
            for i in range(num_clusters):
                sel_index = np.argmin(distances[:, i])
                instance = thisdataset[sel_index]
                ###change tylelabel
                instance[11] = 3
                ###add to mem data
                mem_set[thisrel]['0'].append(instance)  ####positive sample
                cluster_center = kmeans.cluster_centers_[i]
                #print(cluster_center.shape)
                proto_memory[thisrel].append(instance)
        elif selecttype == 1:
            #print("use average embedding")
            samplenum = features.shape[0]
            veclength = features.shape[1]
            sumvec = np.zeros(veclength)
            for j in range(samplenum):
                sumvec += features[j]
            sumvec /= samplenum

            ###find nearest sample
            mindist = 100000000
            minindex = -100
            for j in range(samplenum):
                dist = np.sqrt(np.sum(np.square(features[j] - sumvec)))
                if dist < mindist:
                    minindex = j
                    mindist = dist
            #print(minindex)
            instance = thisdataset[j]
            ###change tylelabel
            instance[11] = 3
            mem_set[thisrel]['0'].append(instance)
            proto_memory[thisrel].append(instance)
        else:
            print("error select type")
    #####to get negative sample  mem_set[thisrel]['1']
    if rela_num > 1:
        ####we need to sample negative samples
        allnegres = {}
        for i in range(rela_num):
            thisnegres = {'h':[],'t':[]}
            currel = current_relations[i]
            thisrelposnum = len(mem_set[currel]['0'])
            #assert thisrelposnum == num_sel_data
            #allnum = list(range(thisrelposnum))
            for j in range(thisrelposnum):
                thisnegres['h'].append(mem_set[currel]['0'][j][3])
                thisnegres['t'].append(mem_set[currel]['0'][j][5])
            allnegres[currel] = thisnegres
        ####get neg sample
        for i in range(rela_num):
            togetnegindex = (i + 1) % rela_num
            togetnegrelname = current_relations[togetnegindex]
            mem_set[current_relations[i]]['1']['h'].extend(allnegres[togetnegrelname]['h'])
            mem_set[current_relations[i]]['1']['t'].extend(allnegres[togetnegrelname]['t'])
    return mem_set

tempthre = 0.2

factorfor2 = 1.0
factorfor3 = 1.0
factorfor4 = 1.0
factorfor5 = 0.1

def train_model_with_hard_neg(config, model, mem_set, traindata, epochs, current_proto, ifnegtive=0):
    print(len(traindata))
    #print(len(train_set))
    mem_data = []
    if len(mem_set) != 0:
        for key in mem_set.keys():
            mem_data.extend(mem_set[key]['0'])
    print(len(mem_data))
    train_set = traindata + mem_data
    #train_set.extend(mem_data)  ########??????maybe some question!! 重复添加mem
    print(len(train_set))

    data_loader = get_data_loader(config, train_set, batch_size=config['batch_size_per_step'])
    model.train()
    criterion = nn.CrossEntropyLoss()
    lossfn = nn.MultiMarginLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    for epoch_i in range(epochs):
        model.set_memorized_prototypes(current_proto)
        losses1 = []
        losses2 = []
        losses3 = []
        losses4 = []
        losses5 = []

        lossesfactor1 = 0.0
        lossesfactor2 = factorfor2
        lossesfactor3 = factorfor3
        lossesfactor4 = factorfor4
        lossesfactor5 = factorfor5
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
                   typelabels) in enumerate(data_loader):
            model.zero_grad()
            #print(len(sentences))
            labels = labels.to(config['device'])
            typelabels = typelabels.to(config['device'])  ####0:rel  1:pos(new train data)  2:neg  3:mem
            numofmem = 0
            numofnewtrain = 0
            allnum = 0
            memindex = []
            for index,onetype in enumerate(typelabels):
                if onetype == 1:
                    numofnewtrain += 1
                if onetype == 3:
                    numofmem += 1
                    memindex.append(index)
                allnum += 1
            #print(numofmem)
            #print(numofnewtrain)
            getnegfromnum = 1
            allneg = []
            alllen = []
            if numofmem > 0:
                ###select neg data for mem
                for oneindex in memindex:
                    negres,lenres = getnegfrombatch(oneindex,firstent,firstentindex,secondent,secondentindex,sentences,lengths,getnegfromnum,allnum,labels,neg_labels)
                    for aa in negres:
                        allneg.append(torch.tensor(aa))
                    for aa in lenres:
                        alllen.append(torch.tensor(aa))
            sentences.extend(allneg)
            lengths.extend(alllen)
            logits, rep = model(sentences, lengths)
            #print(logits.shape)
            #print(rep.shape)
            logits_proto = model.mem_forward(rep)
            #print(logits_proto.shape)
            logitspos = logits[0:allnum,]
            #print(logitspos.shape)
            logits_proto_pos = logits_proto[0:allnum,]
            #print(logits_proto_pos.shape)
            if numofmem > 0:
                logits_proto_neg = logits_proto[allnum:,]

            logits = logitspos
            logits_proto = logits_proto_pos
            loss1 = criterion(logits, labels)
            loss2 = criterion(logits_proto, labels)
            loss4 = lossfn(logits_proto, labels)
            loss3 = torch.tensor(0.0).to(config['device'])
            for index, logit in enumerate(logits):
                score = logits_proto[index]
                preindex = labels[index]
                maxscore = score[preindex]
                size = score.shape[0]
                secondmax = -100000
                for j in range(size):
                    if j != preindex and score[j] > secondmax:
                        secondmax = score[j]
                if secondmax - maxscore + tempthre > 0.0:
                    loss3 += (secondmax - maxscore + tempthre).to(config['device'])
            loss3 /= logits.shape[0]

            start = 0
            loss5 = torch.tensor(0.0).to(config['device'])
            allusenum = 0
            for index in memindex:
                onepos = logits_proto[index]
                posindex = labels[index]
                #poslabelscore = torch.exp(onepos[posindex])
                poslabelscore = onepos[posindex]
                negnum = getnegfromnum * 2
                negscore = torch.tensor(0.0).to(config['device'])
                for ii in range(start, start + negnum):
                    oneneg = logits_proto_neg[ii]
                    #negscore += torch.exp(oneneg[posindex])
                    negscore = oneneg[posindex]
                    if negscore - poslabelscore + 0.01 > 0.0 and negscore < poslabelscore:
                        loss5 += (negscore - poslabelscore + 0.01)
                        allusenum += 1
                #loss5 += (-torch.log(poslabelscore/(poslabelscore+negscore)))
                start += negnum
            #print(len(memindex))
            if len(memindex) == 0:
                loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
            else:
                #loss5 /= len(memindex)
                loss5 = loss5 / allusenum
                #loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4    ###no loss5
                loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4 + loss5 * lossesfactor5    ###with loss5
            loss.backward()
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            losses4.append(loss4.item())
            losses5.append(loss5.item())
            #print("step:\t", step, "\tloss1:\t", loss1.item(), "\tloss2:\t", loss2.item(), "\tloss3:\t", loss3.item(),
            #      "\tloss4:\t", loss4.item(), "\tloss5:\t", loss5.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
        return model

def train_simple_model(config, model, mem_set, train_set, epochs, current_proto, ifusemem=False):
    if ifusemem:
        mem_data = []
        if len(mem_set)!=0:
            for key in mem_set.keys():
                mem_data.extend(mem_set[key]['0'])
        train_set.extend(mem_data)

    data_loader = get_data_loader(config, train_set, batch_size=config['batch_size_per_step'])
    model.train()
    criterion = nn.CrossEntropyLoss()
    lossfn = nn.MultiMarginLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    for epoch_i in range(epochs):
        model.set_memorized_prototypes(current_proto)
        losses1 = []
        losses2 = []
        losses3 = []
        losses4 = []

        lossesfactor1 = 0.0
        lossesfactor2 = factorfor2
        lossesfactor3 = factorfor3
        lossesfactor4 = factorfor4

        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext,
                   lengths, typelabels) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            logits, rep = model(sentences, lengths)
            logits_proto = model.mem_forward(rep)

            labels = labels.to(config['device'])
            loss1 = criterion(logits, labels)
            loss2 = criterion(logits_proto, labels)
            loss4 = lossfn(logits_proto, labels)
            loss3 = torch.tensor(0.0).to(config['device'])
            ###add triple loss
            for index, logit in enumerate(logits):
                score = logits_proto[index]
                preindex = labels[index]
                maxscore = score[preindex]
                size = score.shape[0]
                secondmax = -100000
                for j in range(size):
                    if j != preindex and score[j] > secondmax:
                        secondmax = score[j]
                if secondmax - maxscore + tempthre > 0.0:
                    loss3 += (secondmax - maxscore + tempthre).to(config['device'])

            loss3 /= logits.shape[0]
            loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
            loss.backward()
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            losses4.append(loss4.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
        #print (np.array(losses).mean())
    return model

if __name__ == '__main__':

    select_thredsold_param = 0.65
    select_num = 1
    f = open("config/config_tacred.json", "r")
    config = json.loads(f.read())
    f.close()
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])
    config['neg_sampling'] = False

    root_path = '.'
    word2id = json.load(open(os.path.join(root_path, 'glove/word2id.txt')))
    word2vec = np.load(os.path.join(root_path, 'glove/word2vec.npy'))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    donum = 1
    distantpath = "data/distantdata/"
    file1 = distantpath + "distant.json"
    file2 = distantpath + "exclude_fewrel_distant.json"
    list_data,entpair2scope = process_data(file1,file2)

    topk = 16
    max_sen_length_for_select = 64
    max_sen_lstm_tokenize = 128
    select_thredsold = select_thredsold_param

    print("********* load from ckpt ***********")
    ckptpath = "simmodelckpt"
    print(ckptpath)
    ckpt = torch.load(ckptpath)
    SimModel = BertModel.from_pretrained('bert-base-uncased',state_dict=ckpt["bert-base"]).to(config["device"])

    allunlabledata = np.load("allunlabeldata.npy").astype('float32')
    d = 768 * 2
    index = faiss.IndexFlatIP(d)
    print(index.is_trained)
    index.add(allunlabledata)  # add vectors to the index
    print(index.ntotal)

    for m in range(donum):
        print(m)
        config["rel_cluster_label"] = "data/tacred/CFRLdata_10_100_10_5/rel_cluster_label_" + str(m) + ".npy"
        config['training_file'] = "data/tacred/CFRLdata_10_100_10_5/train_" + str(m) + ".txt"
        config['valid_file'] = "data/tacred/CFRLdata_10_100_10_5/valid_" + str(m) + ".txt"
        config['test_file'] = "data/tacred/CFRLdata_10_100_10_5/test_" + str(m) + ".txt"

        encoderforbase = lstm_encoder(token2id=word2id, word2vec=word2vec, word_size=len(word2vec[0]), max_length=128, pos_size=None,
                                    hidden_size=config['hidden_size'], dropout=0, bidirectional=True, num_layers=1, config=config)
        sampler = data_sampler(config, encoderforbase.tokenizer)
        modelforbase = proto_softmax_layer(encoderforbase, num_class=len(sampler.id2rel), id2rel=sampler.id2rel, drop=0, config=config)
        modelforbase = modelforbase.to(config["device"])

        word2vec_back = word2vec.copy()

        sequence_results = []
        result_whole_test = []
        for i in range(6):

            num_class = len(sampler.id2rel)
            print(config['random_seed'] + 10 * i)
            set_seed(config, config['random_seed'] + 10 * i)
            sampler.set_seed(config['random_seed'] + 10 * i)

            mem_set = {} ####  mem_set = {rel_id:{'0':[positive samples],'1':[negative samples]}} 换5个head 换5个tail
            mem_relations = []   ###not include relation of current task

            past_relations = []

            savetest_all_data = None
            saveseen_relations = []

            proto_memory = []

            for i in range(len(sampler.id2rel)):
                proto_memory.append([sampler.id2rel_pattern[i]])
            oneseqres = []
            ##################################
            whichdataselecct = 1
            ifnorm = True
            ##################################
            for steps, (training_data, valid_data, test_data, test_all_data, seen_relations, current_relations) in enumerate(sampler):
                #print(steps)
                print("------------------------")
                print(len(training_data))
                #for aa in range(20):
                #    print(training_data[aa])
                savetest_all_data = test_all_data
                saveseen_relations = seen_relations

                currentnumber = len(current_relations)
                print(currentnumber)
                print(current_relations)
                divide_train_set = {}
                for relation in current_relations:
                    divide_train_set[relation] = []  ##int
                for data in training_data:
                    divide_train_set[data[0]].append(data)
                print(len(divide_train_set))

                ####select most similar sentence for new task, not for base task

                ####step==0是base model
                if steps == 0:
                    ##train base model
                    print("train base model,not select most similar")

                else:
                    print("train new model,select most similar")
                    selectdata = select_similar_data_new_tac(training_data, tokenizer, entpair2scope, topk,
                                                            max_sen_length_for_select,list_data, config, SimModel,
                                                            select_thredsold,max_sen_lstm_tokenize,encoderforbase.tokenizer,index,ifnorm,select_num)
                    print(len(selectdata))
                    training_data.extend(selectdata)
                    print(len(training_data))
                    #'''

                current_proto = get_memory(config, modelforbase, proto_memory)
                modelforbase = train_simple_model(config, modelforbase, mem_set, training_data, 1,
                                                    current_proto, False)
                select_data(mem_set, proto_memory, config, modelforbase, divide_train_set,
                            config['rel_memory_size'], current_relations, 0)  ##config['rel_memory_size'] == 1

                for j in range(2):
                    current_proto = get_memory(config, modelforbase, proto_memory)
                    modelforbase = train_model_with_hard_neg(config, modelforbase, mem_set, training_data, 1,
                                                                    current_proto, ifnegtive=0)

                current_proto = get_memory(config, modelforbase, proto_memory)
                modelforbase.set_memorized_prototypes(current_proto)
                mem_relations.extend(current_relations)

                currentalltest = []
                for mm in range(len(test_data)):
                    currentalltest.extend(test_data[mm])
                    #eval_model(config, modelforbase, test_data[mm], mem_relations)

                thisstepres = eval_model(config, modelforbase, currentalltest, mem_relations)
                print("step:\t",steps,"\taccuracy:\t",thisstepres)
                oneseqres.append(thisstepres)
            sequence_results.append(np.array(oneseqres))

            #def eval_both_model(config, newmodel, basemodel, test_set, mem_relations, baserelation, newrelation, proto_embed):
            allres = eval_model(config, modelforbase, savetest_all_data, saveseen_relations)
            result_whole_test.append(allres)

            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("after one epoch allres:\t",allres)
            print(result_whole_test)

            # initialize the models
            modelforbase = modelforbase.to('cpu')
            del modelforbase
            gc.collect()
            if config['device'] == 'cuda':
                torch.cuda.empty_cache()
            encoderforbase = lstm_encoder(token2id=word2id, word2vec=word2vec_back.copy(), word_size=len(word2vec[0]),max_length=128, pos_size=None,
                                          hidden_size=config['hidden_size'], dropout=0, bidirectional=True, num_layers=1, config=config)
            modelforbase = proto_softmax_layer(encoderforbase, num_class=len(sampler.id2rel), id2rel=sampler.id2rel,
                                               drop=0, config=config)
            modelforbase.to(config["device"])
            # output the final avg result
        print("Final result!")
        print(result_whole_test)
        for one in sequence_results:
            for item in one:
                sys.stdout.write('%.4f, ' % item)
            print('')
        avg_result_all_test = np.average(sequence_results, 0)
        for one in avg_result_all_test:
            sys.stdout.write('%.4f, ' % one)
        print('')
        print("Finish training............................")
    #'''

