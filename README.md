# [ACL2022] Continual Few-shot Relation Learning via Embedding Space Regularization and Data Augmentation 

The repo is the source code for [Continual Few-shot Relation Learning via Embedding Space Regularization and Data Augmentation](https://openreview.net/forum?id=tN-UlSrCBgM)

Chengwei Qin, Shafiq Joty

Accepted at 60th Annual Meeting of the Association for Computational Linguistics (ACL'22).

## Setup

### 1. Download the code

```
git clone git@github.com:qcwthu/Continual_Fewshot_Relation_Learning.git
cd Continual_Fewshot_Relation_Learning
```

### 2. Install modified transformers

```
cd transformers; pip install .; cd ..
```

### 3. Download pre-trained files

#### 3.1. Pre-trained similarity model

[simmodelckpt](https://drive.google.com/file/d/1zS9gcJOtexA4wenRrKuHcMauEyFZMGG5/view?usp=sharing)

Please name it as 'simmodelckpt' and put it under the root folder of this project.

#### 3.2. Embeddings of all unlabeled data

[allunlabeldata.npy](https://drive.google.com/file/d/1iVi338KPcUPOLPYDY8SwDifhMGaud1kd/view?usp=sharing)

Please name it as 'allunlabeldata.npy' and put it under the root folder of this project.

#### 3.3. Glove embeddings

[glove](https://drive.google.com/drive/folders/1G_D1npBY_nirZGBgWcv7jlZZnYfbxBbn?usp=sharing)

Please put 'glove' folder under the root folder of this project.

#### 3.4 distant.json

[distant.json](https://drive.google.com/file/d/1fj7Z_eRZ37KuHid8a0kwSr2_tXgj3yHy/view?usp=sharing)

### 4. Run script

```
bash runall.sh
```





## Citation

If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.




```
@article{qin2022continual,
  title={Continual Few-shot Relation Learning via Embedding Space Regularization and Data Augmentation},
  author={Qin, Chengwei and Joty, Shafiq},
  journal={arXiv preprint arXiv:2203.02135},
  year={2022}
}
```

