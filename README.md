



原始数据集和代码参考：

> @inproceedings{cao2022c2dsr,
> 	author = {Cao, Jiangxia and Cong, Xin and Sheng, Jiawei and Liu, Tingwen and Wang, Bin},
> 	title = {Contrastive Cross-Domain Sequential Recommendation},
> 	year = {2022},
> 	address = {New York, NY, USA},
> 	url = {https://doi.org/10.1145/3511808.3557262},
> 	doi = {10.1145/3511808.3557262},
> 	booktitle = {Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
> 	pages = {138–147},
> 	numpages = {10},
> 	location = {Atlanta, GA, USA},
> 	series = {CIKM '22}
> }

MGCL数据集参考：

>@inproceedings{Xu2023AMG,
>author = {Xu, Zitao and Pan, Weike and Ming, Zhong},
>title = {A Multi-View Graph Contrastive Learning Framework for Cross-Domain Sequential Recommendation},
>year = {2023},
>address = {New York, NY, USA},
>url = {https://doi.org/10.1145/3604915.3608785},
>booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
>pages = {491–501},
>numpages = {11},
>location = {Singapore, Singapore},
>series = {RecSys '23}
>}



结构：

```
├── dataset
│   ├── Book-CD
│   │   ├── Alist.txt
│   │   ├── Blist.txt
│   │   ├── source_item_id_map.csv
│   │   ├── source_item_text.CLS
│   │   ├── source_item_text.csv
│   │   ├── source_negative.csv
│   │   ├── target_item_id_map.csv
│   │   ├── target_item_text.CLS
│   │   ├── target_item_text.csv
│   │   ├── target_negative.csv
│   │   ├── testdata_new.txt
│   │   ├── traindata_new.txt
│   │   └── validdata_new.txt
		...
		Book-Movie, Movie-CD same to Book-CD
│   ├── Entertainment-Education
│   │   ├── Alist.txt
│   │   ├── Blist.txt
│   │   ├── testdata_new.txt
│   │   ├── traindata_new.txt
│   │   ├── userlist.txt
│   │   └── validdata_new.txt
		...
		Movie-Book, Food-Kitchen 
		...
│   ├── generate_data.py
│   ├── process.py
│   ├── convert2C2DSR.ipynb
│   ├── process_text.ipynb
│   ├── leak_stats.py
│   └── read_file.py
├── logs
├── model
│   ├── C2DSR.py
│   ├── GNN.py
│   └── trainer.py
├── train_rec.py
└── utils
    ├── GraphMaker.py
    ├── helper.py
    ├── loader.py
    └── torch_utils.py
```

数据集：

- 原始数据集：[origin Data](https://github.com/cjx96/C2DSR)

- MGCL数据集：[data](https://csse.szu.edu.cn/staff/panwk/publications/MGCL/)

- 文本描述数据集：[Amazon review data (stanford.edu)](https://snap.stanford.edu/data/amazon/productGraph/)

  ````python
  META_Books http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
  META_CDs http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_CDs_and_Vinyl.json.gz
  META_Movies http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Movies_and_TV.json.gz
  ````

- 文本描述数据集处理文件：process_text.ipynb

- bert模型地址：[bert-base-uncased · Hugging Face](https://huggingface.co/bert-base-uncased)

- 转换MGCL数据集代码：convert2C2DSR.ipynb

- 文本特征提取代码：generate_data.py

数据集处理步骤：

- 对于原始数据集，下载之后可直接使用
- 对于MGCL数据集，需要按如下步骤处理：
  1. 使用dataset中提供的process.py 处理原始的3个数据集
  2. 使用process_text.ipynb 处理Meta数据集
  3. 使用generate_data.py 处理文本描述数据
  4. 使用convert2C2DSR.ipynb 生成转换后的数据集
  5. 确认处理后的文件结构与前面提供的示例一致。

数据处理**完毕**之后，运行脚本，超参数含义在train_rec.py已有说明

````shell
CUDA_VISIBLE_DEVICES=3 python -u train_rec.py --undebug  --hidden_units 128 --id 01 --seed 3407 --batch_size 128 --maxlen 100  --neg_type popular --data_dir Movie-Books > train.log 2>&1&
````



