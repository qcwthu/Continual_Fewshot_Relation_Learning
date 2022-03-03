import sys
import os
import random
import torch
import numpy as np
import re
import json
from collections import defaultdict

import hashlib
def set_seed(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config['n_gpu'] > 0 and torch.cuda.is_available() and config['use_gpu']:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def readtrain(filename):
    f = open(filename,'r')
    res = {}
    while True:
        line = f.readline().strip()
        if not line:
            break
        content = line.split("\t")
        #if len(content) != 7:
        if len(content) != 9:
            print("error!!!")
            exit -1
        else:
            rel = int(content[0])
            if rel not in res:
                res[rel] = []
                res[rel].append(line)
            else:
                res[rel].append(line)
    f.close()
    return res

def transtonpy(data,tokenizer):
    for sample in data:
        tokens = tokenizer.tokenize(sample[2])
        max_length = 128
        length = min(len(tokens), max_length)
        tokens = tokenizer.convert_tokens_to_ids(tokens, unk_id=tokenizer.vocab['[UNK]'])
        if (len(tokens) > max_length):
            tokens = tokens[:max_length]
        sample[2] = tokens
        sample.append(length)
    return np.asarray(data)

def cotinualfewshotpreprocess(config,tokenizer):
    rel_index = np.load("data/fewrel/rel_index.npy")
    #rel_cluster_label = np.load("rel_cluster_label.npy")
    alltraindata = readtrain("data/fewrel/train_all.txt")

    print(len(alltraindata))
    allrel = []
    for i in alltraindata.keys():
        allrel.append(i)
        print(i, "\t", len(alltraindata[i]))
    print(len(allrel))

    alltestdata = readtrain("data/fewrel/test.txt")
    print(len(alltestdata))
    for i in alltestdata.keys():
        print(i, "\t", len(alltestdata[i]))

    samplenum = 1
    basenum = 10
    datanumforbaserel = 100

    allnum = len(allrel)
    waysnum = 10
    howmanyways = (allnum - basenum) // waysnum

    shotnum = 100
    filenum = 9999

    numforeverytestrel = 100
    for i in range(samplenum):
        ####sample basenum base relations
        sample_list = random.sample(allrel, basenum)
        #sample_list = [6, 12, 14, 17, 21, 25, 49, 64, 65, 78]
        print(sample_list)
        ####
        tousetraindata = []
        for k in alltraindata.keys():
            if k in sample_list:
                trainsamplelist = random.sample(alltraindata[k], datanumforbaserel)
                #trainsamplelist = random.sample(alltraindata[k], len(alltraindata[k]))
                tousetraindata.extend(trainsamplelist)
            else:
                trainsamplelist = random.sample(alltraindata[k], shotnum)
                #trainsamplelist = random.sample(alltraindata[k], len(alltraindata[k]))
                tousetraindata.extend(trainsamplelist)
        random.shuffle(tousetraindata)  ###train
        print(len(tousetraindata))

        tousetestdata = []
        for k in alltestdata.keys():
            testsamplelist = random.sample(alltestdata[k], numforeverytestrel)
            #testsamplelist = random.sample(alltestdata[k], len(alltestdata[k]))
            tousetestdata.extend(testsamplelist)
        random.shuffle(tousetestdata)
        print(len(tousetestdata))

        print(rel_index)
        print(sample_list)
        newlabeltasknum = []
        for k in range(allnum):
            if rel_index[k] in sample_list:
                newlabeltasknum.append(howmanyways)
            else:
                newlabeltasknum.append(-1)
        print(newlabeltasknum)
        ###other howmanyways tasks
        temptaskindex = []
        for k in range(howmanyways):
            for j in range(waysnum):
                temptaskindex.append(k)
        random.shuffle(temptaskindex)
        realindex = 0
        for k in range(allnum):
            if newlabeltasknum[k] != -1:
                continue
            else:
                newlabeltasknum[k] = temptaskindex[realindex]
                realindex += 1
        print(realindex)
        print(newlabeltasknum)
        newname = "data/fewrel/CFRLdatatest_10_100_10_"+str(filenum)+"/rel_cluster_label_" + str(i) + ".npy"
        np.save(newname, np.asarray(newlabeltasknum))

        traintxtname = "data/fewrel/CFRLdatatest_10_100_10_"+str(filenum)+"/train_" + str(i) + ".txt"
        fw = open(traintxtname, "w")
        for line in tousetraindata:
            fw.write(line + "\n")
        fw.close()

        testtxtname = "data/fewrel/CFRLdatatest_10_100_10_"+str(filenum)+"/test_" + str(i) + ".txt"
        fw = open(testtxtname, "w")
        for line in tousetestdata:
            fw.write(line + "\n")
        fw.close()

        trainnpyname = "data/fewrel/CFRLdatatest_10_100_10_"+str(filenum)+"/train_" + str(i) + ".npy"
        saveasnpytrain = []
        for l in range(0, len(tousetraindata)):
            items = tousetraindata[l].split("\t")
            relation_ix = int(items[0])
            candidate_ixs = [int(ix) for ix in items[1].split()]
            question = items[2].split('\n')[0]
            saveasnpytrain.append([relation_ix, candidate_ixs, question])
        # print(saveasnpytrain[0])
        tosavetrain = transtonpy(saveasnpytrain, tokenizer)
        np.save(trainnpyname, tosavetrain)

        testnpyname = "data/fewrel/CFRLdatatest_10_100_10_"+str(filenum)+"/test_" + str(i) + ".npy"
        saveasnpytest = []
        for l in range(0, len(tousetestdata)):
            items = tousetestdata[l].split("\t")
            relation_ix = int(items[0])
            candidate_ixs = [int(ix) for ix in items[1].split()]
            question = items[2].split('\n')[0]
            saveasnpytest.append([relation_ix, candidate_ixs, question])
        tosavetest = transtonpy(saveasnpytest, tokenizer)
        np.save(testnpyname, tosavetest)

    newtrain1 = np.load("data/fewrel/CFRLdatatest_10_100_10_"+str(filenum)+"/train_0.npy",allow_pickle=True)
    print(newtrain1.shape)
    print(newtrain1[0])

def getnegfrombatch(oneindex,firstent,firstentindex,secondent,secondentindex,sentences,lengths,getnegfromnum,allnum,labels,neg_labels):
    # thislabel = labels[oneindex]

    ###get information

    thissentence = sentences[oneindex].numpy().tolist()
    #print(thissentence)
    thislength = lengths[oneindex]
    #print(thislength)

    thisfirstent = firstent[oneindex]
    #print(thisfirstent)
    thisfirstentindex = firstentindex[oneindex].numpy().tolist()
    #print(thisfirstentindex)
    headstart = thisfirstentindex[0]
    #print(headstart)
    headend = thisfirstentindex[-1]
    #print(headend)
    posheadlength = len(thisfirstentindex)
    #print(posheadlength)

    thissecondent = secondent[oneindex]
    #print(thissecondent)
    thissecondentindex = secondentindex[oneindex].numpy().tolist()
    #print(thissecondentindex)
    tailstart = thissecondentindex[0]
    #print(tailstart)
    tailend = thissecondentindex[-1]
    #print(tailend)
    postaillength = len(thissecondentindex)
    #print(postaillength)



    negres = []
    lenres = []
    for j in range(getnegfromnum):
        touseindex = (oneindex + j + 1) % allnum
        negusehead = firstent[touseindex].numpy().tolist()
        negheadlength = len(negusehead)
        negusetail = secondent[touseindex].numpy().tolist()
        negtaillength = len(negusetail)
        negsamplechangehead = thissentence[0:headstart] + negusehead + thissentence[headend + 1:]
        changeheadlength = thislength - posheadlength + negheadlength

        negsamplechangetail = thissentence[0:tailstart] + negusetail + thissentence[tailend + 1:]
        changetaillength = thislength - postaillength + negtaillength

        #######get 2
        negres.append(negsamplechangehead)
        lenres.append(changeheadlength)
        negres.append(negsamplechangetail)
        lenres.append(changetaillength)

        ######get 1

    return np.asarray(negres),np.asarray(lenres)

def getnegfrombatchnew(oneindex,firstent,firstentindex,secondent,secondentindex,sentences,lengths,getnegfromnum,allnum,labels,neg_labels):
    # thislabel = labels[oneindex]

    ###get information

    thissentence = sentences[oneindex].numpy().tolist()
    #print(thissentence)
    thislength = lengths[oneindex]
    #print(thislength)

    thisfirstent = firstent[oneindex]
    #print(thisfirstent)
    thisfirstentindex = firstentindex[oneindex].numpy().tolist()
    #print(thisfirstentindex)
    headstart = thisfirstentindex[0]
    #print(headstart)
    headend = thisfirstentindex[-1]
    #print(headend)
    posheadlength = len(thisfirstentindex)
    #print(posheadlength)

    thissecondent = secondent[oneindex]
    #print(thissecondent)
    thissecondentindex = secondentindex[oneindex].numpy().tolist()
    #print(thissecondentindex)
    tailstart = thissecondentindex[0]
    #print(tailstart)
    tailend = thissecondentindex[-1]
    #print(tailend)
    postaillength = len(thissecondentindex)
    #print(postaillength)



    negres = []
    lenres = []
    for j in range(getnegfromnum):
        touseindex = (oneindex + j + 1) % allnum
        negusehead = firstent[touseindex].numpy().tolist()
        negheadlength = len(negusehead)
        negusetail = secondent[touseindex].numpy().tolist()
        negtaillength = len(negusetail)
        negsamplechangehead = thissentence[0:headstart] + negusehead + thissentence[headend + 1:]
        changeheadlength = thislength - posheadlength + negheadlength

        negsamplechangetail = thissentence[0:tailstart] + negusetail + thissentence[tailend + 1:]
        changetaillength = thislength - postaillength + negtaillength

        #######get 1
        aa = random.randint(0,1)
        if aa == 1:
            negres.append(negsamplechangehead)
            lenres.append(changeheadlength)
        else:
            negres.append(negsamplechangetail)
            lenres.append(changetaillength)


    return np.asarray(negres),np.asarray(lenres)

def getnegfrombatch_bert(oneindex,firstent,firstentindex,secondent,secondentindex,sentences,lengths,getnegfromnum,allnum,labels,neg_labels,config):
    thissentence = sentences[oneindex].cpu().numpy().tolist()
    thislength = lengths[oneindex]
    thisfirstent = firstent[oneindex]
    thisfirstentindex = firstentindex[oneindex].numpy().tolist()
    headstart = thisfirstentindex[0]
    headend = thisfirstentindex[-1]
    posheadlength = len(thisfirstentindex)

    thissecondent = secondent[oneindex]
    thissecondentindex = secondentindex[oneindex].numpy().tolist()
    tailstart = thissecondentindex[0]
    tailend = thissecondentindex[-1]
    postaillength = len(thissecondentindex)

    negres = []
    maskres = []
    for j in range(getnegfromnum):
        touseindex = (oneindex + j + 1) % allnum
        negusehead = firstent[touseindex].numpy().tolist()
        negheadlength = len(negusehead)
        negusetail = secondent[touseindex].numpy().tolist()
        negtaillength = len(negusetail)
        negsamplechangehead = thissentence[0:headstart] + negusehead + thissentence[headend + 1:]
        changeheadlength = thislength - posheadlength + negheadlength
        if len(negsamplechangehead) > config["max_length"]:
            negsamplechangehead = negsamplechangehead[0:config["max_length"]]
        for i in range(len(negsamplechangehead), config["max_length"]):
            negsamplechangehead.append(0)
        mask1 = []
        for i in range(0, changeheadlength):
            mask1.append(1)
        for i in range(changeheadlength, config["max_length"]):
            mask1.append(0)
        if len(mask1) > config["max_length"]:
            mask1 = mask1[0:config["max_length"]]

        negsamplechangetail = thissentence[0:tailstart] + negusetail + thissentence[tailend + 1:]
        changetaillength = thislength - postaillength + negtaillength
        if len(negsamplechangetail) > config["max_length"]:
            negsamplechangetail = negsamplechangetail[0:config["max_length"]]
        for i in range(len(negsamplechangetail), config["max_length"]):
            negsamplechangetail.append(0)
        mask2 = []
        for i in range(0, changetaillength):
            mask2.append(1)
        for i in range(changetaillength, config["max_length"]):
            mask2.append(0)
        if len(mask2) > config["max_length"]:
            mask2 = mask2[0:config["max_length"]]

        if len(mask1) != len(mask2):
            print(len(mask1))
            print(len(mask2))
            print(mask1)
            print(mask2)

        negres.append(negsamplechangehead)
        maskres.append(mask1)
        negres.append(negsamplechangetail)
        maskres.append(mask2)

    return np.asarray(negres),np.asarray(maskres)


def getnegforonerel(mem_set,key,neg_mem_data):
    negusehead = mem_set[key]['1']['h'][0]
    negheadlength = len(negusehead)
    negusetail = mem_set[key]['1']['t'][0]
    negtaillength = len(negusetail)

    possen = mem_set[key]['0'][0][2]  ####positive sentence tokens
    poslen = mem_set[key]['0'][0][7]

    poshead = mem_set[key]['0'][0][3]
    posheadindex = mem_set[key]['0'][0][4]
    headstart = posheadindex[0]
    headend = posheadindex[-1]
    posheadlength = len(posheadindex)

    postail = mem_set[key]['0'][0][5]
    postailindex = mem_set[key]['0'][0][6]
    tailstart = postailindex[0]
    tailend = postailindex[-1]
    postaillength = len(postailindex)

    negsamplechangehead = possen[0:headstart] + negusehead + possen[headend + 1:]
    changeheadlength = poslen - posheadlength + negheadlength

    negsamplechangetail = possen[0:tailstart] + negusetail + possen[tailend + 1:]
    changetaillength = poslen - postaillength + negtaillength

    newnegsample1 = []
    newnegsample1.append(mem_set[key]['0'][0][0])
    newnegsample1.append(mem_set[key]['0'][0][1])
    newnegsample1.append(negsamplechangehead)
    newnegsample1.append(negusehead)
    newnegsample1.append(posheadindex)  ####wrong index
    newnegsample1.append(postail)
    newnegsample1.append(postailindex)
    newnegsample1.append("neghead")
    newnegsample1.append("postail")
    newnegsample1.append("fakesen")
    newnegsample1.append(changeheadlength)

    newnegsample1.append(2)

    newnegsample2 = []
    newnegsample2.append(mem_set[key]['0'][0][0])
    newnegsample2.append(mem_set[key]['0'][0][1])
    newnegsample2.append(negsamplechangetail)
    newnegsample2.append(poshead)
    newnegsample2.append(posheadindex)
    newnegsample2.append(negusetail)
    newnegsample2.append(postailindex)
    newnegsample1.append("poshead")
    newnegsample1.append("negtail")
    newnegsample1.append("fakesen")
    newnegsample2.append(changetaillength)
    newnegsample2.append(2)

    # print(newnegsample2)

    neg_mem_data.append(np.asarray(newnegsample1))
    neg_mem_data.append(np.asarray(newnegsample2))

def getposandneg(logits,logits_proto,labels,typelabels):
    numofpos = 0
    numofneg = 0
    for index, logit in enumerate(logits):
        type = typelabels[index]
        if type == 1:
            numofpos += 1
        else:
            numofneg += 1
    embedlen = logits.shape[1]

    tensorpos = torch.zeros((numofpos, embedlen))
    protopos = torch.zeros((numofpos, embedlen))
    poslabels = torch.zeros([numofpos],dtype=torch.long)

    tensorneg = torch.zeros((numofneg, embedlen))
    protoneg = torch.zeros((numofneg, embedlen))
    neglabels = torch.zeros([numofneg],dtype=torch.long)

    posindex = 0
    negindex = 0

    for index, logit in enumerate(logits):
        type = typelabels[index]
        if type == 1:
            tensorpos[posindex] = logits[index]
            protopos[posindex] = logits_proto[index]
            poslabels[posindex] = labels[index]
            posindex += 1
        else:
            tensorneg[negindex] = logits[index]
            protoneg[negindex] = logits_proto[index]
            neglabels[negindex] = labels[index]
            negindex += 1
    #numofpos
    #numofneg
    #print("numofpos:\t",numofpos,"numofneg:\t",numofneg)
    return tensorpos,protopos,poslabels,tensorneg,protoneg,neglabels,numofneg

def handletoken(raw_text,h_pos_li,t_pos_li,tokenizer):
    h_pattern = re.compile("\* h \*")
    t_pattern = re.compile("\^ t \^")
    err = 0
    tokens = []
    h_mention = []
    t_mention = []
    raw_text_list = raw_text.split(" ")
    for i, token in enumerate(raw_text_list):
        token = token.lower()
        if i >= h_pos_li[0] and i <= h_pos_li[-1]:
            if i == h_pos_li[0]:
                tokens += ['*', 'h', '*']
            h_mention.append(token)
            continue
        if i >= t_pos_li[0] and i <= t_pos_li[-1]:
            if i == t_pos_li[0]:
                tokens += ['^', 't', '^']
            t_mention.append(token)
            continue
        tokens.append(token)
    text = " ".join(tokens)
    h_mention = " ".join(h_mention)
    t_mention = " ".join(t_mention)
    #print(text)
    #print(h_mention)
    #print(t_mention)
    tokenized_text = tokenizer.tokenize(text)
    tokenized_head = tokenizer.tokenize(h_mention)
    tokenized_tail = tokenizer.tokenize(t_mention)

    p_text = " ".join(tokenized_text)
    p_head = " ".join(tokenized_head)
    p_tail = " ".join(tokenized_tail)
    p_text = h_pattern.sub("[unused0] " + p_head + " [unused1]", p_text)
    p_text = t_pattern.sub("[unused2] " + p_tail + " [unused3]", p_text)
    #print(p_text)
    f_text = ("[CLS] " + p_text + " [SEP]").split()
    #print(f_text)
    # If h_pos_li and t_pos_li overlap, we can't find head entity or tail entity.
    try:
        h_pos = f_text.index("[unused0]")
    except:
        err += 1
        h_pos = 0
    try:
        t_pos = f_text.index("[unused2]")
    except:
        err += 1
        t_pos = 0

    tokenized_input = tokenizer.convert_tokens_to_ids(f_text)

    return tokenized_input, h_pos, t_pos


def filter_sentence(sentence):
    head_pos = sentence["h"]["pos"][0]
    tail_pos = sentence["t"]["pos"][0]

    if sentence["h"]["name"] == sentence["t"]["name"]:  # head mention equals tail mention
        return True

    if head_pos[0] >= tail_pos[0] and head_pos[0] <= tail_pos[-1]:  # head mentioin and tail mention overlap
        return True

    if tail_pos[0] >= head_pos[0] and tail_pos[0] <= head_pos[-1]:  # head mentioin and tail mention overlap
        return True

    return False

def process_data(file1,file2):
    data1 = json.load(open(file1))
    #data2 = json.load(open(file2))
    data2 = {}
    max_num = 16 ###max number for every entity pair
    ent_data = defaultdict(list)
    for key in data1.keys():
        for sentence in data1[key]:
            if filter_sentence(sentence):
                continue
            head = sentence["h"]["id"]
            tail = sentence["t"]["id"]
            newsen = sentence
            #print(newsen["tokens"])
            newtokens = " ".join(newsen["tokens"]).lower().split(" ")
            #print(newtokens)
            newsen["tokens"] = newtokens
            #print(newsen)
            ent_data[head + "#" + tail].append(newsen)
    for key in data2.keys():
        for sentence in data2[key]:
            if filter_sentence(sentence):
                continue
            head = sentence["h"]["id"]
            tail = sentence["t"]["id"]
            newsen = sentence
            newtokens = " ".join(newsen["tokens"]).lower().split(" ")
            newsen["tokens"] = newtokens
            ent_data[head + "#" + tail].append(newsen)
    ll = 0
    list_data = []
    entpair2scope = {}
    for key in ent_data.keys():
        #if len(ent_data[key]) < 2:
        #    continue
        list_data.extend(ent_data[key][0:max_num])
        entpair2scope[key] = [ll, len(list_data)]
        ll = len(list_data)
    return list_data,entpair2scope

def select_similar_data_new(training_data,tokenizer,entpair2scope,topk,max_sen_length_for_select,list_data,config,SimModel,select_thredsold,max_sen_lstm_tokenize,enctokenizer,faissindex,ifnorm,select_num=2):

    #use both methods
    selectdata = []
    alladdnum = 0
    #md5 = hashlib.md5()
    has = 0
    nothas = 0
    for onedata in training_data:
        label = onedata[0]
        text = onedata[9]
        headid = onedata[7]
        tailid = onedata[8]
        headindex = onedata[4]
        tailindex = onedata[6]
        onedatatoken, onedatahead, onedatatail = handletoken(text, headindex, tailindex, tokenizer)

        onedicid = headid + "#" + tailid
        tmpselectnum = 0
        if onedicid in entpair2scope:
            #print("bbbbbbbbbbbbbbb")
            has += 1
            thispairnum = entpair2scope[onedicid][1] - entpair2scope[onedicid][0]
            #if thispairnum > topk:
            if True:
                ###choose topk
                alldisforthispair = []
                input_ids = np.zeros((thispairnum + 1, max_sen_length_for_select), dtype=int)
                mask = np.zeros((thispairnum + 1, max_sen_length_for_select), dtype=int)
                h_pos = np.zeros((thispairnum + 1), dtype=int)
                t_pos = np.zeros((thispairnum + 1), dtype=int)
                for index in range(entpair2scope[onedicid][0], entpair2scope[onedicid][1]):
                    oneres = list_data[index]
                    tokens = " ".join(oneres["tokens"])
                    ###sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1]
                    '''
                    sentence example:
                    {
                        'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                        'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                        't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                        'r': 'P1'
                    }
                    '''
                    hposstart = oneres["h"]["pos"][0][0]
                    hposend = oneres["h"]["pos"][0][-1]
                    tposstart = oneres["t"]["pos"][0][0]
                    tposend = oneres["t"]["pos"][0][-1]
                    tokenres, headpos, tailpos = handletoken(tokens, [hposstart, hposend], [tposstart, tposend],
                                                             tokenizer)
                    length = min(len(tokenres), max_sen_length_for_select)
                    input_ids[index - entpair2scope[onedicid][0]][0:length] = tokenres[0:length]
                    mask[index - entpair2scope[onedicid][0]][0:length] = 1
                    h_pos[index - entpair2scope[onedicid][0]] = min(headpos, max_sen_length_for_select - 1)
                    t_pos[index - entpair2scope[onedicid][0]] = min(tailpos, max_sen_length_for_select - 1)
                # onedatatoken, onedatahead, onedatatail
                length = min(len(onedatatoken), max_sen_length_for_select)
                input_ids[thispairnum][0:length] = onedatatoken[0:length]
                mask[thispairnum][0:length] = 1
                h_pos[thispairnum] = min(onedatahead, max_sen_length_for_select - 1)
                t_pos[thispairnum] = min(onedatatail, max_sen_length_for_select - 1)
                ###cal score
                # print(input_ids)
                # print(mask)
                input_ids = torch.from_numpy(input_ids).to(config["device"])
                mask = torch.from_numpy(mask).to(config["device"])
                h_pos = torch.from_numpy(h_pos).to(config["device"])
                t_pos = torch.from_numpy(t_pos).to(config["device"])
                outputs = SimModel(input_ids, mask)
                indice = torch.arange(input_ids.size()[0])
                h_state = outputs[0][indice, h_pos]
                t_state = outputs[0][indice, t_pos]
                state = torch.cat((h_state, t_state), 1)
                # print(state.shape)
                query = state[thispairnum, :].view(1, state.shape[-1])
                toselect = state[0:thispairnum, :].view(thispairnum, state.shape[-1])
                if ifnorm:
                    #print("norm")
                    querynorm = query / query.norm(dim=1)[:, None]
                    toselectnorm = toselect / toselect.norm(dim=1)[:, None]
                    res = (querynorm * toselectnorm).sum(-1)
                    #print(res)
                else:
                    res = (query * toselect).sum(-1)
                # print(res)
                pred = []
                for i in range(res.size(0)):
                    pred.append((res[i], i))
                pred.sort(key=lambda x: x[0], reverse=True)
                # print(pred)
                # print(res.shape)
                # print(res)
                ####select from pred
                selectedindex = []
                tmpselectnum = 0
                prescore= -100.0
                for k in range(len(pred)):
                    thistext = " ".join(list_data[entpair2scope[onedicid][0] + pred[k][1]]["tokens"])
                    if thistext == text:
                        continue
                    #if tmpselectnum < topk and pred[k][0] > select_thredsold and pred[k][0] != prescore:
                    if tmpselectnum < topk and pred[k][0] > select_thredsold:
                        selectedindex.append(pred[k][1])
                        prescore = pred[k][0]
                        tmpselectnum += 1
                #print("tmpselectnum: ",tmpselectnum)
                for onenum in selectedindex:
                    onelabel = label
                    oneneg = [label]
                    onesen = " ".join(list_data[entpair2scope[onedicid][0] + onenum]["tokens"])
                    tokens = enctokenizer.tokenize(onesen)
                    length = min(len(tokens), max_sen_lstm_tokenize)
                    tokens = enctokenizer.convert_tokens_to_ids(tokens, unk_id=enctokenizer.vocab['[UNK]'])
                    if (len(tokens) > max_sen_lstm_tokenize):
                        tokens = tokens[:max_sen_lstm_tokenize]
                    fakefirstent = [554, 555]
                    fakefirstindex = [0, 1]
                    fakesecondent = [665, 666]
                    fakesecondindex = [3, 4]
                    fakeheadid = "fheadid"
                    faketailid = "ftailid"
                    fakerawtext = "fakefake"
                    typelabel = 1  ###positive sample
                    oneseldata = [onelabel, oneneg, tokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex, fakeheadid, faketailid, fakerawtext, length, typelabel]
                    selectdata.append(np.asarray(oneseldata))
                    #selectres.append(list_data[entpair2scope[onedicid][0] + onenum])
                alladdnum += tmpselectnum
        else:
            #print("nothing!continue")
            #continue
            #print("hghagdhasdgjahsgdjahgdjahgdjasgdj")
        #if onedicid not in entpair2scope or tmpselectnum == 0:
            #print("aaaaaaaaaa")
            nothas += 1
            # print("not in! use fasis")
            topuse = select_num
            # faissindex
            input_ids = np.zeros((1, max_sen_length_for_select), dtype=int)
            mask = np.zeros((1, max_sen_length_for_select), dtype=int)
            h_pos = np.zeros((1), dtype=int)
            t_pos = np.zeros((1), dtype=int)
            length = min(len(onedatatoken), max_sen_length_for_select)
            input_ids[0][0:length] = onedatatoken[0:length]
            mask[0][0:length] = 1
            h_pos[0] = min(onedatahead, max_sen_length_for_select - 1)
            t_pos[0] = min(onedatatail, max_sen_length_for_select - 1)

            input_ids = torch.from_numpy(input_ids).to(config["device"])
            mask = torch.from_numpy(mask).to(config["device"])
            h_pos = torch.from_numpy(h_pos).to(config["device"])
            t_pos = torch.from_numpy(t_pos).to(config["device"])
            outputs = SimModel(input_ids, mask)
            indice = torch.arange(input_ids.size()[0])
            h_state = outputs[0][indice, h_pos]
            t_state = outputs[0][indice, t_pos]
            state = torch.cat((h_state, t_state), 1)
            # print(state.shape)
            #####some problems, need normalize!!!!!!!!!!!!
            if ifnorm:
                state = state / state.norm(dim=1)[:, None]
            ########################################
            query = state.view(1, state.shape[-1]).cpu().detach().numpy()

            D, I = faissindex.search(query, topuse)
            newtouse = topuse
            newadd = 0
            for i in range(newtouse):
                thisdis = D[0][i]
                #print("&&&&&&&&&&&&&&&&&&")
                #print(thisdis)
                ###whether to use this?
                #if thisdis < 0.95:
                #    continue
                newadd += 1
                onenum = I[0][i]
                onelabel = label
                oneneg = [label]
                onesen = " ".join(list_data[onenum]["tokens"])
                ###handle onesen
                onesen.replace("\n\n\n", " ")
                onesen.replace("\n\n", " ")
                onesen.replace("\n", " ")
                #print(text)
                #print("********************************")
                #print(onesen)
                #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                tokens = enctokenizer.tokenize(onesen)
                length = min(len(tokens), max_sen_lstm_tokenize)
                tokens = enctokenizer.convert_tokens_to_ids(tokens, unk_id=enctokenizer.vocab['[UNK]'])
                if (len(tokens) > max_sen_lstm_tokenize):
                    tokens = tokens[:max_sen_lstm_tokenize]
                fakefirstent = [554, 555]
                fakefirstindex = [0, 1]
                fakesecondent = [665, 666]
                fakesecondindex = [3, 4]
                fakeheadid = "fheadid"
                faketailid = "ftailid"
                fakerawtext = "fakefake"
                typelabel = 1  ###positive sample
                oneseldata = [onelabel, oneneg, tokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex,
                              fakeheadid, faketailid, fakerawtext, length, typelabel]
                selectdata.append(np.asarray(oneseldata))
            alladdnum += newadd
    return selectdata

def select_similar_data_new_bert(training_data,tokenizer,entpair2scope,topk,max_sen_length_for_select,list_data,config,SimModel,select_thredsold,max_sen_lstm_tokenize,enctokenizer,faissindex,ifnorm,select_num=2):
    selectdata = []
    alladdnum = 0
    #md5 = hashlib.md5()
    has = 0
    nothas = 0
    for onedata in training_data:
        label = onedata[0]
        text = onedata[9]
        headid = onedata[7]
        tailid = onedata[8]
        headindex = onedata[4]
        tailindex = onedata[6]
        onedatatoken, onedatahead, onedatatail = handletoken(text, headindex, tailindex, tokenizer)

        onedicid = headid + "#" + tailid
        tmpselectnum = 0
        if onedicid in entpair2scope:
            #print("bbbbbbbbbbbbbbb")
            has += 1
            thispairnum = entpair2scope[onedicid][1] - entpair2scope[onedicid][0]
            #if thispairnum > topk:
            if True:
                ###choose topk
                alldisforthispair = []
                input_ids = np.zeros((thispairnum + 1, max_sen_length_for_select), dtype=int)
                mask = np.zeros((thispairnum + 1, max_sen_length_for_select), dtype=int)
                h_pos = np.zeros((thispairnum + 1), dtype=int)
                t_pos = np.zeros((thispairnum + 1), dtype=int)
                for index in range(entpair2scope[onedicid][0], entpair2scope[onedicid][1]):
                    oneres = list_data[index]
                    tokens = " ".join(oneres["tokens"])
                    ###sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1]
                    '''
                    sentence example:
                    {
                        'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                        'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                        't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                        'r': 'P1'
                    }
                    '''
                    hposstart = oneres["h"]["pos"][0][0]
                    hposend = oneres["h"]["pos"][0][-1]
                    tposstart = oneres["t"]["pos"][0][0]
                    tposend = oneres["t"]["pos"][0][-1]
                    tokenres, headpos, tailpos = handletoken(tokens, [hposstart, hposend], [tposstart, tposend],
                                                             tokenizer)
                    length = min(len(tokenres), max_sen_length_for_select)
                    input_ids[index - entpair2scope[onedicid][0]][0:length] = tokenres[0:length]
                    mask[index - entpair2scope[onedicid][0]][0:length] = 1
                    h_pos[index - entpair2scope[onedicid][0]] = min(headpos, max_sen_length_for_select - 1)
                    t_pos[index - entpair2scope[onedicid][0]] = min(tailpos, max_sen_length_for_select - 1)
                # onedatatoken, onedatahead, onedatatail
                length = min(len(onedatatoken), max_sen_length_for_select)
                input_ids[thispairnum][0:length] = onedatatoken[0:length]
                mask[thispairnum][0:length] = 1
                h_pos[thispairnum] = min(onedatahead, max_sen_length_for_select - 1)
                t_pos[thispairnum] = min(onedatatail, max_sen_length_for_select - 1)
                ###cal score
                # print(input_ids)
                # print(mask)
                input_ids = torch.from_numpy(input_ids).to(config["device"])
                mask = torch.from_numpy(mask).to(config["device"])
                h_pos = torch.from_numpy(h_pos).to(config["device"])
                t_pos = torch.from_numpy(t_pos).to(config["device"])
                outputs = SimModel(input_ids, mask)
                indice = torch.arange(input_ids.size()[0])
                h_state = outputs[0][indice, h_pos]
                t_state = outputs[0][indice, t_pos]
                state = torch.cat((h_state, t_state), 1)
                # print(state.shape)
                query = state[thispairnum, :].view(1, state.shape[-1])
                toselect = state[0:thispairnum, :].view(thispairnum, state.shape[-1])
                if ifnorm:
                    #print("norm")
                    querynorm = query / query.norm(dim=1)[:, None]
                    toselectnorm = toselect / toselect.norm(dim=1)[:, None]
                    res = (querynorm * toselectnorm).sum(-1)
                    #print(res)
                else:
                    res = (query * toselect).sum(-1)
                # print(res)
                pred = []
                for i in range(res.size(0)):
                    pred.append((res[i], i))
                pred.sort(key=lambda x: x[0], reverse=True)
                # print(pred)
                # print(res.shape)
                # print(res)
                ####select from pred
                selectedindex = []
                tmpselectnum = 0
                prescore= -100.0
                for k in range(len(pred)):
                    thistext = " ".join(list_data[entpair2scope[onedicid][0] + pred[k][1]]["tokens"])
                    if thistext == text:
                        continue
                    #if tmpselectnum < topk and pred[k][0] > select_thredsold and pred[k][0] != prescore:
                    if tmpselectnum < topk and pred[k][0] > select_thredsold:
                        selectedindex.append(pred[k][1])
                        prescore = pred[k][0]
                        tmpselectnum += 1
                #print("tmpselectnum: ",tmpselectnum)
                for onenum in selectedindex:
                    oneres = list_data[entpair2scope[onedicid][0] + onenum]
                    onelabel = label
                    oneneg = [label]
                    onesen = " ".join(oneres["tokens"])
                    hposstart = oneres["h"]["pos"][0][0]
                    hposend = oneres["h"]["pos"][0][-1]
                    tposstart = oneres["t"]["pos"][0][0]
                    tposend = oneres["t"]["pos"][0][-1]
                    tokens, headpos, tailpos = handletoken(onesen, [hposstart, hposend], [tposstart, tposend],
                                                             tokenizer)


                    length = min(len(tokens), max_sen_lstm_tokenize)
                    if (len(tokens) > max_sen_lstm_tokenize):
                        tokens = tokens[:max_sen_lstm_tokenize]
                    newtokens = []
                    for i in range(0, length):
                        newtokens.append(tokens[i])
                    for i in range(length, max_sen_lstm_tokenize):
                        newtokens.append(0)
                    fakefirstent = [554, 555]
                    fakefirstindex = [0, 1]
                    fakesecondent = [665, 666]
                    fakesecondindex = [3, 4]
                    fakeheadid = "fheadid"
                    faketailid = "ftailid"
                    fakerawtext = "fakefake"
                    typelabel = 1  ###positive sample
                    mask = []
                    for i in range(0, length):
                        mask.append(1)
                    for i in range(length, max_sen_lstm_tokenize):
                        mask.append(0)
                    oneseldata = [onelabel, oneneg, newtokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex, fakeheadid, faketailid, fakerawtext, length, typelabel, mask]
                    selectdata.append(np.asarray(oneseldata))
                    #selectres.append(list_data[entpair2scope[onedicid][0] + onenum])
                alladdnum += tmpselectnum
        #else:
        if onedicid not in entpair2scope or tmpselectnum == 0:
            #print("aaaaaaaaaa")
            nothas += 1
            # print("not in! use fasis")
            topuse = select_num
            # faissindex
            input_ids = np.zeros((1, max_sen_length_for_select), dtype=int)
            mask = np.zeros((1, max_sen_length_for_select), dtype=int)
            h_pos = np.zeros((1), dtype=int)
            t_pos = np.zeros((1), dtype=int)
            length = min(len(onedatatoken), max_sen_length_for_select)
            input_ids[0][0:length] = onedatatoken[0:length]
            mask[0][0:length] = 1
            h_pos[0] = min(onedatahead, max_sen_length_for_select - 1)
            t_pos[0] = min(onedatatail, max_sen_length_for_select - 1)

            input_ids = torch.from_numpy(input_ids).to(config["device"])
            mask = torch.from_numpy(mask).to(config["device"])
            h_pos = torch.from_numpy(h_pos).to(config["device"])
            t_pos = torch.from_numpy(t_pos).to(config["device"])
            outputs = SimModel(input_ids, mask)
            indice = torch.arange(input_ids.size()[0])
            h_state = outputs[0][indice, h_pos]
            t_state = outputs[0][indice, t_pos]
            state = torch.cat((h_state, t_state), 1)
            # print(state.shape)
            #####some problems, need normalize!!!!!!!!!!!!
            if ifnorm:
                state = state / state.norm(dim=1)[:, None]
            ########################################
            query = state.view(1, state.shape[-1]).cpu().detach().numpy()

            D, I = faissindex.search(query, topuse)
            newtouse = topuse
            newadd = 0
            for i in range(newtouse):
                thisdis = D[0][i]
                #print("&&&&&&&&&&&&&&&&&&")
                #print(thisdis)
                ###whether to use this?
                #if thisdis < 0.95:
                #    continue
                newadd += 1
                onenum = I[0][i]
                onelabel = label
                oneneg = [label]
                oneres = list_data[onenum]
                onesen = " ".join(oneres["tokens"])
                hposstart = oneres["h"]["pos"][0][0]
                hposend = oneres["h"]["pos"][0][-1]
                tposstart = oneres["t"]["pos"][0][0]
                tposend = oneres["t"]["pos"][0][-1]
                tokens, headpos, tailpos = handletoken(onesen, [hposstart, hposend], [tposstart, tposend],
                                                       tokenizer)

                length = min(len(tokens), max_sen_lstm_tokenize)
                if (len(tokens) > max_sen_lstm_tokenize):
                    tokens = tokens[:max_sen_lstm_tokenize]
                newtokens = []
                for i in range(0, length):
                    newtokens.append(tokens[i])
                for i in range(length, max_sen_lstm_tokenize):
                    newtokens.append(0)
                fakefirstent = [554, 555]
                fakefirstindex = [0, 1]
                fakesecondent = [665, 666]
                fakesecondindex = [3, 4]
                fakeheadid = "fheadid"
                faketailid = "ftailid"
                fakerawtext = "fakefake"
                typelabel = 1  ###positive sample
                mask = []
                for i in range(0, length):
                    mask.append(1)
                for i in range(length, max_sen_lstm_tokenize):
                    mask.append(0)
                oneseldata = [onelabel, oneneg, newtokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex,
                              fakeheadid, faketailid, fakerawtext, length, typelabel,mask]
                selectdata.append(np.asarray(oneseldata))
            alladdnum += newadd
    return selectdata


def select_similar_data_new_tac(training_data,tokenizer,entpair2scope,topk,max_sen_length_for_select,list_data,config,SimModel,select_thredsold,max_sen_lstm_tokenize,enctokenizer,faissindex,ifnorm,select_num=2):
    selectdata = []
    alladdnum = 0
    #md5 = hashlib.md5()
    has = 0
    nothas = 0
    for onedata in training_data:
        label = onedata[0]
        text = onedata[9]
        headid = onedata[7]
        tailid = onedata[8]
        headindex = onedata[4]
        tailindex = onedata[6]
        onedatatoken, onedatahead, onedatatail = handletoken(text, headindex, tailindex, tokenizer)

        onedicid = headid + "#" + tailid
        tmpselectnum = 0
        if onedicid in entpair2scope:
            #print("bbbbbbbbbbbbbbb")
            has += 1
            thispairnum = entpair2scope[onedicid][1] - entpair2scope[onedicid][0]
            #if thispairnum > topk:
            if True:
                ###choose topk
                alldisforthispair = []
                input_ids = np.zeros((thispairnum + 1, max_sen_length_for_select), dtype=int)
                mask = np.zeros((thispairnum + 1, max_sen_length_for_select), dtype=int)
                h_pos = np.zeros((thispairnum + 1), dtype=int)
                t_pos = np.zeros((thispairnum + 1), dtype=int)
                for index in range(entpair2scope[onedicid][0], entpair2scope[onedicid][1]):
                    oneres = list_data[index]
                    tokens = " ".join(oneres["tokens"])
                    ###sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1]
                    '''
                    sentence example:
                    {
                        'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                        'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                        't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                        'r': 'P1'
                    }
                    '''
                    hposstart = oneres["h"]["pos"][0][0]
                    hposend = oneres["h"]["pos"][0][-1]
                    tposstart = oneres["t"]["pos"][0][0]
                    tposend = oneres["t"]["pos"][0][-1]
                    tokenres, headpos, tailpos = handletoken(tokens, [hposstart, hposend], [tposstart, tposend],
                                                             tokenizer)
                    length = min(len(tokenres), max_sen_length_for_select)
                    input_ids[index - entpair2scope[onedicid][0]][0:length] = tokenres[0:length]
                    mask[index - entpair2scope[onedicid][0]][0:length] = 1
                    h_pos[index - entpair2scope[onedicid][0]] = min(headpos, max_sen_length_for_select - 1)
                    t_pos[index - entpair2scope[onedicid][0]] = min(tailpos, max_sen_length_for_select - 1)
                # onedatatoken, onedatahead, onedatatail
                length = min(len(onedatatoken), max_sen_length_for_select)
                input_ids[thispairnum][0:length] = onedatatoken[0:length]
                mask[thispairnum][0:length] = 1
                h_pos[thispairnum] = min(onedatahead, max_sen_length_for_select - 1)
                t_pos[thispairnum] = min(onedatatail, max_sen_length_for_select - 1)
                ###cal score
                # print(input_ids)
                # print(mask)
                input_ids = torch.from_numpy(input_ids).to(config["device"])
                mask = torch.from_numpy(mask).to(config["device"])
                h_pos = torch.from_numpy(h_pos).to(config["device"])
                t_pos = torch.from_numpy(t_pos).to(config["device"])
                outputs = SimModel(input_ids, mask)
                indice = torch.arange(input_ids.size()[0])
                h_state = outputs[0][indice, h_pos]
                t_state = outputs[0][indice, t_pos]
                state = torch.cat((h_state, t_state), 1)
                # print(state.shape)
                query = state[thispairnum, :].view(1, state.shape[-1])
                toselect = state[0:thispairnum, :].view(thispairnum, state.shape[-1])
                if ifnorm:
                    #print("norm")
                    querynorm = query / query.norm(dim=1)[:, None]
                    toselectnorm = toselect / toselect.norm(dim=1)[:, None]
                    res = (querynorm * toselectnorm).sum(-1)
                    #print(res)
                else:
                    res = (query * toselect).sum(-1)
                # print(res)
                pred = []
                for i in range(res.size(0)):
                    pred.append((res[i], i))
                pred.sort(key=lambda x: x[0], reverse=True)
                # print(pred)
                # print(res.shape)
                # print(res)
                ####select from pred
                selectedindex = []
                tmpselectnum = 0
                prescore= -100.0
                for k in range(len(pred)):
                    thistext = " ".join(list_data[entpair2scope[onedicid][0] + pred[k][1]]["tokens"])
                    if thistext == text:
                        continue
                    #if tmpselectnum < topk and pred[k][0] > select_thredsold and pred[k][0] != prescore:
                    if tmpselectnum < topk and pred[k][0] > select_thredsold:
                        selectedindex.append(pred[k][1])
                        prescore = pred[k][0]
                        tmpselectnum += 1
                #print("tmpselectnum: ",tmpselectnum)
                for onenum in selectedindex:
                    onelabel = label
                    oneneg = [label]
                    onesen = " ".join(list_data[entpair2scope[onedicid][0] + onenum]["tokens"])
                    tokens = enctokenizer.tokenize(onesen)
                    length = min(len(tokens), max_sen_lstm_tokenize)
                    tokens = enctokenizer.convert_tokens_to_ids(tokens, unk_id=enctokenizer.vocab['[UNK]'])
                    if (len(tokens) > max_sen_lstm_tokenize):
                        tokens = tokens[:max_sen_lstm_tokenize]
                    fakefirstent = [554, 555]
                    fakefirstindex = [0, 1]
                    fakesecondent = [665, 666]
                    fakesecondindex = [3, 4]
                    fakeheadid = "fheadid"
                    faketailid = "ftailid"
                    fakerawtext = "fakefake"
                    typelabel = 1  ###positive sample
                    oneseldata = [onelabel, oneneg, tokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex, fakeheadid, faketailid, fakerawtext, length, typelabel]
                    selectdata.append(np.asarray(oneseldata))
                    #selectres.append(list_data[entpair2scope[onedicid][0] + onenum])
                alladdnum += tmpselectnum
        #else:
        if onedicid not in entpair2scope or tmpselectnum == 0:
            #print("aaaaaaaaaa")
            nothas += 1
            # print("not in! use fasis")
            topuse = select_num
            # faissindex
            input_ids = np.zeros((1, max_sen_length_for_select), dtype=int)
            mask = np.zeros((1, max_sen_length_for_select), dtype=int)
            h_pos = np.zeros((1), dtype=int)
            t_pos = np.zeros((1), dtype=int)
            length = min(len(onedatatoken), max_sen_length_for_select)
            input_ids[0][0:length] = onedatatoken[0:length]
            mask[0][0:length] = 1
            h_pos[0] = min(onedatahead, max_sen_length_for_select - 1)
            t_pos[0] = min(onedatatail, max_sen_length_for_select - 1)

            input_ids = torch.from_numpy(input_ids).to(config["device"])
            mask = torch.from_numpy(mask).to(config["device"])
            h_pos = torch.from_numpy(h_pos).to(config["device"])
            t_pos = torch.from_numpy(t_pos).to(config["device"])
            outputs = SimModel(input_ids, mask)
            indice = torch.arange(input_ids.size()[0])
            h_state = outputs[0][indice, h_pos]
            t_state = outputs[0][indice, t_pos]
            state = torch.cat((h_state, t_state), 1)
            # print(state.shape)
            #####some problems, need normalize!!!!!!!!!!!!
            if ifnorm:
                state = state / state.norm(dim=1)[:, None]
            ########################################
            query = state.view(1, state.shape[-1]).cpu().detach().numpy()

            D, I = faissindex.search(query, topuse)
            newtouse = topuse
            newadd = 0
            for i in range(newtouse):
                thisdis = D[0][i]
                #print("&&&&&&&&&&&&&&&&&&")
                #print(thisdis)
                ###whether to use this?
                if thisdis < 0.96:
                    continue
                newadd += 1
                onenum = I[0][i]
                onelabel = label
                oneneg = [label]
                onesen = " ".join(list_data[onenum]["tokens"])
                ###handle onesen
                onesen.replace("\n\n\n", " ")
                onesen.replace("\n\n", " ")
                onesen.replace("\n", " ")
                #print(text)
                #print("********************************")
                #print(onesen)
                #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                tokens = enctokenizer.tokenize(onesen)
                length = min(len(tokens), max_sen_lstm_tokenize)
                tokens = enctokenizer.convert_tokens_to_ids(tokens, unk_id=enctokenizer.vocab['[UNK]'])
                if (len(tokens) > max_sen_lstm_tokenize):
                    tokens = tokens[:max_sen_lstm_tokenize]
                fakefirstent = [554, 555]
                fakefirstindex = [0, 1]
                fakesecondent = [665, 666]
                fakesecondindex = [3, 4]
                fakeheadid = "fheadid"
                faketailid = "ftailid"
                fakerawtext = "fakefake"
                typelabel = 1  ###positive sample
                oneseldata = [onelabel, oneneg, tokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex,
                              fakeheadid, faketailid, fakerawtext, length, typelabel]
                selectdata.append(np.asarray(oneseldata))
            alladdnum += newadd
    return selectdata

def select_similar_data_new_bert_tac(training_data,tokenizer,entpair2scope,topk,max_sen_length_for_select,list_data,config,SimModel,select_thredsold,max_sen_lstm_tokenize,enctokenizer,faissindex,ifnorm,select_num=2):
    selectdata = []
    alladdnum = 0
    #md5 = hashlib.md5()
    has = 0
    nothas = 0
    for onedata in training_data:
        label = onedata[0]
        text = onedata[9]
        headid = onedata[7]
        tailid = onedata[8]
        headindex = onedata[4]
        tailindex = onedata[6]
        onedatatoken, onedatahead, onedatatail = handletoken(text, headindex, tailindex, tokenizer)

        onedicid = headid + "#" + tailid
        tmpselectnum = 0
        if onedicid in entpair2scope:
            #print("bbbbbbbbbbbbbbb")
            has += 1
            thispairnum = entpair2scope[onedicid][1] - entpair2scope[onedicid][0]
            #if thispairnum > topk:
            if True:
                ###choose topk
                alldisforthispair = []
                input_ids = np.zeros((thispairnum + 1, max_sen_length_for_select), dtype=int)
                mask = np.zeros((thispairnum + 1, max_sen_length_for_select), dtype=int)
                h_pos = np.zeros((thispairnum + 1), dtype=int)
                t_pos = np.zeros((thispairnum + 1), dtype=int)
                for index in range(entpair2scope[onedicid][0], entpair2scope[onedicid][1]):
                    oneres = list_data[index]
                    tokens = " ".join(oneres["tokens"])
                    ###sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1]
                    '''
                    sentence example:
                    {
                        'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                        'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                        't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                        'r': 'P1'
                    }
                    '''
                    hposstart = oneres["h"]["pos"][0][0]
                    hposend = oneres["h"]["pos"][0][-1]
                    tposstart = oneres["t"]["pos"][0][0]
                    tposend = oneres["t"]["pos"][0][-1]
                    tokenres, headpos, tailpos = handletoken(tokens, [hposstart, hposend], [tposstart, tposend],
                                                             tokenizer)
                    length = min(len(tokenres), max_sen_length_for_select)
                    input_ids[index - entpair2scope[onedicid][0]][0:length] = tokenres[0:length]
                    mask[index - entpair2scope[onedicid][0]][0:length] = 1
                    h_pos[index - entpair2scope[onedicid][0]] = min(headpos, max_sen_length_for_select - 1)
                    t_pos[index - entpair2scope[onedicid][0]] = min(tailpos, max_sen_length_for_select - 1)
                # onedatatoken, onedatahead, onedatatail
                length = min(len(onedatatoken), max_sen_length_for_select)
                input_ids[thispairnum][0:length] = onedatatoken[0:length]
                mask[thispairnum][0:length] = 1
                h_pos[thispairnum] = min(onedatahead, max_sen_length_for_select - 1)
                t_pos[thispairnum] = min(onedatatail, max_sen_length_for_select - 1)
                ###cal score
                # print(input_ids)
                # print(mask)
                input_ids = torch.from_numpy(input_ids).to(config["device"])
                mask = torch.from_numpy(mask).to(config["device"])
                h_pos = torch.from_numpy(h_pos).to(config["device"])
                t_pos = torch.from_numpy(t_pos).to(config["device"])
                outputs = SimModel(input_ids, mask)
                indice = torch.arange(input_ids.size()[0])
                h_state = outputs[0][indice, h_pos]
                t_state = outputs[0][indice, t_pos]
                state = torch.cat((h_state, t_state), 1)
                # print(state.shape)
                query = state[thispairnum, :].view(1, state.shape[-1])
                toselect = state[0:thispairnum, :].view(thispairnum, state.shape[-1])
                if ifnorm:
                    #print("norm")
                    querynorm = query / query.norm(dim=1)[:, None]
                    toselectnorm = toselect / toselect.norm(dim=1)[:, None]
                    res = (querynorm * toselectnorm).sum(-1)
                    #print(res)
                else:
                    res = (query * toselect).sum(-1)
                # print(res)
                pred = []
                for i in range(res.size(0)):
                    pred.append((res[i], i))
                pred.sort(key=lambda x: x[0], reverse=True)
                # print(pred)
                # print(res.shape)
                # print(res)
                ####select from pred
                selectedindex = []
                tmpselectnum = 0
                prescore= -100.0
                for k in range(len(pred)):
                    thistext = " ".join(list_data[entpair2scope[onedicid][0] + pred[k][1]]["tokens"])
                    if thistext == text:
                        continue
                    #if tmpselectnum < topk and pred[k][0] > select_thredsold and pred[k][0] != prescore:
                    if tmpselectnum < topk and pred[k][0] > select_thredsold:
                        selectedindex.append(pred[k][1])
                        prescore = pred[k][0]
                        tmpselectnum += 1
                #print("tmpselectnum: ",tmpselectnum)
                for onenum in selectedindex:
                    oneres = list_data[entpair2scope[onedicid][0] + onenum]
                    onelabel = label
                    oneneg = [label]
                    onesen = " ".join(oneres["tokens"])
                    hposstart = oneres["h"]["pos"][0][0]
                    hposend = oneres["h"]["pos"][0][-1]
                    tposstart = oneres["t"]["pos"][0][0]
                    tposend = oneres["t"]["pos"][0][-1]
                    tokens, headpos, tailpos = handletoken(onesen, [hposstart, hposend], [tposstart, tposend],
                                                             tokenizer)


                    length = min(len(tokens), max_sen_lstm_tokenize)
                    if (len(tokens) > max_sen_lstm_tokenize):
                        tokens = tokens[:max_sen_lstm_tokenize]
                    newtokens = []
                    for i in range(0, length):
                        newtokens.append(tokens[i])
                    for i in range(length, max_sen_lstm_tokenize):
                        newtokens.append(0)
                    fakefirstent = [554, 555]
                    fakefirstindex = [0, 1]
                    fakesecondent = [665, 666]
                    fakesecondindex = [3, 4]
                    fakeheadid = "fheadid"
                    faketailid = "ftailid"
                    fakerawtext = "fakefake"
                    typelabel = 1  ###positive sample
                    mask = []
                    for i in range(0, length):
                        mask.append(1)
                    for i in range(length, max_sen_lstm_tokenize):
                        mask.append(0)
                    oneseldata = [onelabel, oneneg, newtokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex, fakeheadid, faketailid, fakerawtext, length, typelabel, mask]
                    selectdata.append(np.asarray(oneseldata))
                    #selectres.append(list_data[entpair2scope[onedicid][0] + onenum])
                alladdnum += tmpselectnum
        #else:
        if onedicid not in entpair2scope or tmpselectnum == 0:
            #print("aaaaaaaaaa")
            nothas += 1
            # print("not in! use fasis")
            topuse = select_num
            # faissindex
            input_ids = np.zeros((1, max_sen_length_for_select), dtype=int)
            mask = np.zeros((1, max_sen_length_for_select), dtype=int)
            h_pos = np.zeros((1), dtype=int)
            t_pos = np.zeros((1), dtype=int)
            length = min(len(onedatatoken), max_sen_length_for_select)
            input_ids[0][0:length] = onedatatoken[0:length]
            mask[0][0:length] = 1
            h_pos[0] = min(onedatahead, max_sen_length_for_select - 1)
            t_pos[0] = min(onedatatail, max_sen_length_for_select - 1)

            input_ids = torch.from_numpy(input_ids).to(config["device"])
            mask = torch.from_numpy(mask).to(config["device"])
            h_pos = torch.from_numpy(h_pos).to(config["device"])
            t_pos = torch.from_numpy(t_pos).to(config["device"])
            outputs = SimModel(input_ids, mask)
            indice = torch.arange(input_ids.size()[0])
            h_state = outputs[0][indice, h_pos]
            t_state = outputs[0][indice, t_pos]
            state = torch.cat((h_state, t_state), 1)
            # print(state.shape)
            #####some problems, need normalize!!!!!!!!!!!!
            if ifnorm:
                state = state / state.norm(dim=1)[:, None]
            ########################################
            query = state.view(1, state.shape[-1]).cpu().detach().numpy()

            D, I = faissindex.search(query, topuse)
            newtouse = topuse
            newadd = 0
            for i in range(newtouse):
                thisdis = D[0][i]
                #print("&&&&&&&&&&&&&&&&&&")
                #print(thisdis)
                ###whether to use this?
                if thisdis < 0.96:
                    continue
                newadd += 1
                onenum = I[0][i]
                onelabel = label
                oneneg = [label]
                oneres = list_data[onenum]
                onesen = " ".join(oneres["tokens"])
                hposstart = oneres["h"]["pos"][0][0]
                hposend = oneres["h"]["pos"][0][-1]
                tposstart = oneres["t"]["pos"][0][0]
                tposend = oneres["t"]["pos"][0][-1]
                tokens, headpos, tailpos = handletoken(onesen, [hposstart, hposend], [tposstart, tposend],
                                                       tokenizer)

                length = min(len(tokens), max_sen_lstm_tokenize)
                if (len(tokens) > max_sen_lstm_tokenize):
                    tokens = tokens[:max_sen_lstm_tokenize]
                newtokens = []
                for i in range(0, length):
                    newtokens.append(tokens[i])
                for i in range(length, max_sen_lstm_tokenize):
                    newtokens.append(0)
                fakefirstent = [554, 555]
                fakefirstindex = [0, 1]
                fakesecondent = [665, 666]
                fakesecondindex = [3, 4]
                fakeheadid = "fheadid"
                faketailid = "ftailid"
                fakerawtext = "fakefake"
                typelabel = 1  ###positive sample
                mask = []
                for i in range(0, length):
                    mask.append(1)
                for i in range(length, max_sen_lstm_tokenize):
                    mask.append(0)
                oneseldata = [onelabel, oneneg, newtokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex,
                              fakeheadid, faketailid, fakerawtext, length, typelabel,mask]
                selectdata.append(np.asarray(oneseldata))
            alladdnum += newadd
    return selectdata