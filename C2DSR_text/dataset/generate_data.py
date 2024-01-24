import os 
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
import random
from transformers import AutoModel, AutoTokenizer
import re
import argparse



def generate_item_embedding(args, item_text_df, tokenizer, model):
    print(f'Generate Text Embedding by {args.emb_type}: ')
    print(' Dataset: ', args.dataset)

    # order by the new id to generate the embedding
    items_text_list =item_text_df["text"].tolist()
    print("test the item text list is right ", items_text_list[0])
    embeddings = []
    start, batch_size = 0, 16
    # 解释：start是一个索引，batch_size是一个批次的大小
    while start < len(items_text_list):
        
        sentences = items_text_list[start: start + batch_size]
        
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors='pt').to(args.device)
        
        outputs = model(**encoded_sentences)
        if args.emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
            embeddings.append(cls_output)
        elif args.emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:,1:,:].sum(dim=1) / \
                encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
        start += batch_size
    # obtain all the embedding 
    # then use the cat to concat the embedding
    # the shape of embeddings is (item_num, 768)
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)


    file = os.path.join(args.output_path,args.output_name + "." +args.emb_type)
    # save to file by use the torch save
    embeddings.tofile(file)
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='target_item_text', help='the dataset name')
    parser.add_argument('--input_path', type=str, default='/home/temp_user/yanjie/C2DSR/C2DSR_src/dataset/Movie-Book/')
    parser.add_argument('--output_path', type=str, default='/home/temp_user/yanjie/C2DSR/C2DSR_src/dataset/Movie-Book/')
    parser.add_argument('--output_name', type=str, default='target_item_text')
    parser.add_argument('--gpu_id', type=int, default=2, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='/home/temp_user/yanjie/MGCL/bert_base_uncased')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    return parser.parse_args()


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def load_plm(model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


if __name__ == '__main__':
    args = parse_args()
    
    # train test valid has split
    
    device = set_device(args.gpu_id)
    args.device = device
    plm_tokenizer, plm_model = load_plm(args.plm_name)
    plm_model = plm_model.to(device)


    # create output dir
    check_path(args.output_path)

    # load the text info
    text_info = pd.read_csv(args.input_path + args.dataset + '.csv', sep='\t')

    print("load the text infomation success!")    
    # generate PLM emb and save to file
    
    generate_item_embedding(args, text_info, plm_tokenizer, plm_model)
    
    