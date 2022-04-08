# blackboxsok

Below, we list the candidate experiments to run in each domain and also the corresponding source code.

## Selection Criteria

We only select attacks that: 1) reported state-of-the-art (SOTA) results in the paper with enough baselines, 2) peer-reviewed or have open source implementations available. Note that there can exist multiple SOTA attacks as those SOTA methods are not individually compared. In that case, we include all of them.

## Image Domain

Black-box attacks are heavily investigated in image domain and hence, we can summarize and syatemize existing knowledge and also gain new insights. Below are the list of experiments to run. The candidate attacks are selected to reflect the current state-of-the-art in each black-box attack scenario.

### Hard-label, No Access, No-/Partial-/Full Auxiliary Information

1. Changes to the inputs, gradient or loss function w.r.t. source models: All of these works assume the same or similar training data, but we can also check their performance on some different datasets, which covers `partial-/full-auxiliary information` categories. Currently, no work is done for `no-auxiliary` information category.
    * [On Success and Simplicity:
A Second Look at Transferable Targeted Attacks](https://proceedings.neurips.cc/paper/2021/file/30d454f09b771b9f65e3eaf6e00fa7bd-Paper.pdf), [Code](https://github.com/ZhengyuZhao/Targeted-Tansfer): `loss function` - uses logits loss instead of traditional cross entropy loss and uses larger iterations. Seems to help more with targeted attack, but not untargeted attacks.
    * [A Little Robustness Goes a Long Way: Leveraging Robust Features for Targeted Transfer Attacks](https://proceedings.neurips.cc/paper/2021/file/50f3f8c42b998a48057e9d33f4144b8b-Paper.pdf), [No Code]: `source model type` - published at NeurIPS 2021, no code is available, but should be easy to implement. Main message from the paper is, slightly robust source models show better transferability to unknown target models, once the local attacks are using with the attack listed above.
    * [Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation](https://openreview.net/forum?id=i7FNvHnPvPc) [No Code]: `loss function` - Unpublished and no code available. Main messsage is that, instead of minimizing loss of single adversarial example, we should seek adversarial examples locating at the low-value and flat region of the loss landscape.
    * [Staircase Sign Method for Boosting Adversarial Attacks](https://arxiv.org/abs/2104.09722), [Code](https://github.com/qilong-zhang/Staircase-sign-method): `gradient computation` - unpublished, main message is, either using gradient sign or full gradient value is suboptimal for transferability. We should use some intermediate gradients between sign and the exact value.
    * [Boosting Adversarial Transferability through Enhanced Momentum](https://arxiv.org/pdf/2103.10609.pdf), [No Code]: `gradient computation` enhances the momentum further for better transferability.
    * [Adaptive Image Transformations for Transfer-based Adversarial Attack](https://arxiv.org/abs/2111.13844), [No Code]: `Input` - Adaptive input transformation for better transferability.
    * [Improving Adversarial Transferability with Spatial Momentum](https://arxiv.org/abs/2203.13479), [No Code]: `Gradient Computation` - spatial momentum method for boosted transferability.
    * [Exploring Transferable and Robust Adversarial Perturbation Generation from the Perspective of Network Hierarchy](https://arxiv.org/abs/2108.07033), [No Code]: `loss function` - the main message is, the intermediate layers are more helpful than the final output layer, which is somewhat similar to the resource intensive transfer attacks based on intermediate features.
2. Requires additional model (binary classifier, generator, some pretrained models):
    * [Data-Free Adversarial Perturbations for Practical Black-Box Attacks](https://arxiv.org/pdf/2003.01295.pdf), [No Code]:  Published at ICCV 2021, requires some pretrained model on different dataset, which has no overlap with the data of target models (originally designed for `partial auxiliary information`). We can extend this to full information case by using pretrained models on the same dataset.
    * [Adversarial Attack across Datasets](https://arxiv.org/pdf/2110.07718.pdf), [No Code]: Unpublished, requires training a meta-learner based training on a different dataset withgout label space overlap with the data of target models. Extension to full-auxiliary information is straightforward.
    * [Perturbing Across the Feature Hierarchy to Improve Standard and Strict Blackbox Attack Transferability](https://arxiv.org/pdf/2004.14861.pdf), [No Code]: NeurIPS 2020, requires training binary classifier that learns the target class distribution of the intermediate layers of source network. For `partial-auxiliary information` setting, the training data have partial overlap or no overlap, but the label space overlaps.
    * [Can Targeted Adversarial Examples Transfer When the Source and Target Models Have No Label Space Overlap?](https://arxiv.org/pdf/2103.09916.pdf), [No Code]: similar idea as above. However, the considered setting is, there is even no label space overlap and aims to find the good proxy class (of the target class in mind) for the source models.
    * [On Generating Transferable Targeted Perturbations](), [Code](): Published at ICCV 2021, requires training a generator for target class feature distribution. This method is outperformed by the first attack of this list in the targeted setting and is also designed only for targeted setting. However, it may still be interesting to see if it will perform better for partial auxiliary information.

## Malware Domain

### Hard-label, No Access, No-/Partial-/Full Auxiliary Information

1. Requires no access to training data or target model architecture
    * [Adversarial EXEmples: A Survey and Experimental Evaluation of Practical Attacks on Machine Learning for Windows Malware Detection](https://arxiv.org/pdf/2008.07125.pdf), [Code](https://github.com/pralab/secml_malware): `genetic algorothm` `transfer` - Analysis of Windows malware. Proposes 3 attacks that outperform existing ones in white-box (and use for transfer) and black-box scenarios. The black-box variant is the simplest & tries to minimize the loss with a surrogate model. The paper does not seem to have a core taxonomy, but rather shows how existing attacks can be implemented with RAMEN (their framework). Attack bounded by total number of queries.

2. Does not explicitly use training data, but the data used for training local models is labeld by one of the target models (VirusTotal)
    * [Toward an Effective Black-Box Adversarial Attack on Functional JavaScript Malware against Commercial Anti-Virus](https://dl.acm.org/doi/pdf/10.1145/3459637.3481956) , No Code: `loss function` - Attacks against Javascript Malware detection systems. Full black-box scenario with unlimited query access. Train substitute model (using term frequency as features), with a PGD + seq2seq attack on the proxy model. PGD is launched on the term frequencies, followed by building adversarial vocabulary. Injection code is sampled (for the best variant) using a seq2seq model.