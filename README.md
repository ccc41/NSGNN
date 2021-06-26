# Probing Negative Sampling for Contrastive Learning to Learn Graph Representations
This is the code for our ECML-PKDD 2021 paper "[Probing Negative Sampling for Contrastive Learning to Learn Graph Representations](https://arxiv.org/abs/2104.06317)", which is the first proposals of node-wise contrastive learning to graph data embedding. We also address the class collision problem of the contrastive learning and distorted results caused by imbalanced distribution of the dataset.
## Usage
#### Dependencies

#### Train 
Train the model with the specified dataset
```
python train.py --dataset
```
## Cite
If you find this work is useful, please cite the following:
```
@inproceedings{chen2021probing,
      title={Probing Negative Sampling for Contrastive Learning to Learn Graph Representations}, 
      author={Shiyi Chen and Ziao Wang and Xinni Zhang and Xiaofeng Zhang and Dan Peng},
      booktitle = {Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases},
      year={2021}
}
```
