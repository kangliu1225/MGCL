# MGCL
## Multimodal Graph Contrastive Learning for Recommendation

![framework of MEGCF](model.jpg)

Many state-of-the-art multimedia recommender efforts effectively alleviate the issues of sparsity and cold-start via modeling multimodal user preference. The core paradigm of them is based on graph learning techniques, which perform high-order message propagation of multimodal information on  user-item interaction graph, so as to obtain the node representations that contain both interactive- and multimodal-dimension user preferences. However, we argue that such a paradigm is suboptimal because it ignores two problems, including (1) the presence of a large number of preference-independent noisy data in the items' multimodal content, and (2) the propagation of this multimodal noise over the interaction graph contaminates the representations of both interactive- and multimodal-dimension user preferences.

In this work, we aim to reduce the negative effects of multimodal noise and further improve user preference modeling. Towards this end, we develop a multimodal graph contrastive learning (MGCL) approach, which decomposes user preferences into multiple dimensions and performs cross-dimension mutual information maximization, so that user preference modeling over different dimensions can be enhanced with each other. In particular, we first adopt the graph learning approach to generate representations of users and items in the interaction and multimodal dimensions, respectively. Then, we construct an additional contrastive learning task to maximize the consistency between different dimensions. Extensive experiments on three public datasets validate the effectiveness and scalability of the proposed MGCL.

We provide tensorflow implementation for MEGCF. **we rerun the MEGCF code on the three datasets and record the results in Model/Log/result-Art(or beauty, Taobao).txt.**

## specific results:
- **amazon-beauty:**best result- hr@5,10,20=0.5589368258859132,0.6614663585001644,0.7605932203388645,ndcg@5,10,20=0.4346333457935099,0.4678397335140145,0.49290631919252564
- **Art:** best result- hr@5,10,20=0.7105901053047834,0.7908603218753782,0.8627458772101337,ndcg@5,10,20=0.6106452104466733,0.6366201960701804,0.654870696013297
- **Taobao:**


## prerequisites

- Tensorflow 1.10.0
- Python 3.5
- NVIDIA GPU + CUDA + CuDNN
