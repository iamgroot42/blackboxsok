# blackboxsok

Below, we list the candidate experiments to run in each domain and also the corresponding source code.

## Selection Criteria and Other Information

We only select attacks that: 1) reported state-of-the-art (SOTA) results in the paper with enough baselines, 2) peer-reviewed or have open source implementations available. Note that there can exist multiple SOTA attacks as those SOTA methods are not individually compared. In that case, we include all of them. We plan to report experiments for: 1) 1-2 standard models (undefended) and 1-2 robust models (can be obtained from robust bench: https://robustbench.github.io), 2) targeted (with different target classes) and untargeted settings, 2) different attack strenths.

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
    * [On Generating Transferable Targeted Perturbations](https://arxiv.org/abs/2103.14641), [Code](https://github.com/Muzammal-Naseer/TTP): Published at ICCV 2021, requires training a generator for target class feature distribution. This method is outperformed by the first attack of this list in the targeted setting and is also designed only for targeted setting. However, it may still be interesting to see if it will perform better for partial auxiliary information. 

### Full-Score, Limited/Unlimited Access, No-/Partial-/Full Auxiliary Information
1. No auxiliary information: 
    * [Square Attack: a query-efficient black-box adversarial attack via random search](https://arxiv.org/abs/1912.00049) [Code](https://github.com/max-andr/square-attack): still the SOTA without additional auxiliary information. 
    * [Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks](https://arxiv.org/abs/2006.12834), [Code](https://github.com/fra31/sparse-rs): current SOTA for L_0 norm.
    * [BayesOpt Adversarial Attack](https://openreview.net/forum?id=Hkem-lrtvH), [Code](https://github.com/rubinxin/BayesOpt_Attack): Publisghed at ICLR 2020, mainly designed for low quqery regime and need to check if it performs better than `square-attack`.
2. Partial auxiliary information:
    * [Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks](https://arxiv.org/abs/2111.01714), [Code](https://github.com/boschresearch/meta-rs): Published at NeurIPS 2021, meta-learns the search distribution of square-attack and improves its performance compared to manual tuning. 
    * not many attacks for this cases, but can check any of the methods in full-score category can be used to derive some attacks in this setting. 
3. Full auxiliary information: 
    * Test the combination of best transfer attacks and best no auxiliary information attacks above and see if we can get SOTA performance. 
    * [Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks](https://arxiv.org/abs/2111.01714), [Code](https://github.com/boschresearch/meta-rs): Published at NeurIPS 2021, meta-learns the search distribution of square-attack and improves its performance compared to manual tuning. 
    * [Diversity can be Transferred: Output Diversification for White- and Black-box Attacks](https://arxiv.org/abs/2003.06878), [Code](https://github.com/ermongroup/ODS): Published at NeurIPS 2020, uses diversified sampling distribution from source models to boost SimBA attack. * [Simulating Unknown Target Models for Query-Efficient Black-box Attacks](https://arxiv.org/abs/2009.00960), [Code](https://github.com/machanic/SimulatorAttack): Published at CVPR 2021 2020, simulates target model and trains better source models along the black-box query process. 
    * [Learning Black-Box Attackers with Transferable Priors and Query Feedback](https://arxiv.org/abs/2010.11742), [Code](https://github.com/TrustworthyDL/LeBA): NeurIPS 2021, hybrid attack with tunable local models using higher order gradients. 
    * [QueryNet: Attack by Multi-Identity Surrogates](https://arxiv.org/abs/2105.15010), [Code](https://github.com/allenchen1998/querynet): smarter way of utilizing multiple surrogate models to provide better starting points for black-box attacks and also tunes local models in the attack process.  
    * [Boosting Black-Box Attack with Partially Transferred Conditional Adversarial Distribution](https://arxiv.org/abs/2006.08538), [No Code]: Unpublished, the results are relatively comprehensive and if we can get the code, it is worthwhile to play. 
    * [Black-Box Adversarial Attack with Transferable Model-based Embedding](https://arxiv.org/abs/1911.07140), [Code](https://github.com/TransEmbedBA/TREMBA) (ICLR 2020): uses the latent space representation as the starting point and deloys NES attack subsequently (only applies to gradient estimation attacks). 
    * [Query-efficient Meta Attack to Deep Neural Networks](https://openreview.net/forum?id=Skxd6gSYDS), [Code](https://github.com/dydjw9/MetaAttack_ICLR2020): Puboished at ICLR 2020, uses meta-learning to extract attacks from prior attacks, and then leverages the attack pattern in the actual black-box attacks. 

### Hard-label, No Access, No-/Partial-/Full Auxiliary Information
1. No auxiliary information:
    * [RayS: A Ray Searching Method for Hard-label Adversarial Attack](https://arxiv.org/abs/2006.12792), [code](https://github.com/uclaml/RayS): SOTA attack with no auxiliary information for L_infty setting. We will need to adapt this attack to targeted setting also, the original paper only does for untargeted setting. Applicable to Hybrid attack.   
    * [Query Efficient Decision Based Sparse Attacks Against Black-Box Deep Learning Models](https://arxiv.org/abs/2202.00091), [No Code]: reported SOTA in L_0 setting. Applicable to hybrid attack. 
    * [Hard-label Manifolds: Unexpected Advantages of Query Efficiency for Finding On-manifold Adversarial Examples](https://arxiv.org/abs/2103.03325), [No Code]: Possible new insights (double check paper details), from UFL security group.
    * [Simple and Efficient Hard Label Black-box Adversarial Attacks in Low Query Budget Regimes](https://arxiv.org/abs/2007.07210), [Code]( https://github.com/satyanshukla/bayes_attack): Bayesian optimization, suits better for limited access scenarios.
2. Partial auxiliary information: 
    * No existing work, but can combine like hybrid attack with No Access+Partial Information setting to propose new attacks for this category. Easy to implement. 
3. Full auxiliary information: 
    * Combine listed attacks in No-Auxiliary information with techniques found in full-auxiliary information+No-access category. 
    * TODO: check whether the tunable local model hybrid attacks in full-score setting is applicable here.
    * [Progressive-Scale Boundary Blackbox Attack via Projective Gradient Estimation](https://arxiv.org/abs/2106.06056), [Code](https://github.com/AI-secure/PSBA): requires training an autoencoder (latent space projection) on the same train data (applicable to hybrid attack). 

### Top-K, Limited/Unlimited Access, No-/Partial-/Full Auxiliary Information
1. No Auxiliary information:
    * [NES attack](https://arxiv.org/abs/1804.08598), [Code](https://github.com/labsix/limited-blackbox-attacks): The only paper that considers Top-k prediction settings. Their main idea for top-k case should also be applicable to other attacks in full-score case. 
    * Adapt the [BayesOpt Adversarial Attack](https://openreview.net/forum?id=Hkem-lrtvH)([Code](https://github.com/rubinxin/BayesOpt_Attack)) using the the attack strategy in `NES` attack and check if it gives better performace, especially in limited access setting.
    * Adapt the [Square Attack: a query-efficient black-box adversarial attack via random search](https://arxiv.org/abs/1912.00049) [Code](https://github.com/max-andr/square-attack) using the idea in `NES`.
2. Partial auxiliary information: adapt the [Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks](https://arxiv.org/abs/2111.01714), [Code](https://github.com/boschresearch/meta-rs), while the meta-training distribution is now searched on a different dataset compared to the dataset of the target model. This experiments should be enough for this category.
3. Full auxiliary information: [Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks](https://arxiv.org/abs/2111.01714), [Code](https://github.com/boschresearch/meta-rs), and the meta-training distribution is obtained on the exact same dataset of the target model. This experiments should be enough for this category.

## Malware Domain

### Hard-label, No Access, No-/Partial-/Full Auxiliary Information

1. Requires no access to training data or target model architecture
    * [Adversarial EXEmples: A Survey and Experimental Evaluation of Practical Attacks on Machine Learning for Windows Malware Detection](https://arxiv.org/pdf/2008.07125.pdf), [Code](https://github.com/pralab/secml_malware): `genetic algorothm` `transfer` - Analysis of Windows malware. Proposes 3 attacks that outperform existing ones in white-box (and use for transfer) and black-box scenarios. The black-box variant is the simplest & tries to minimize the loss with a surrogate model. The paper does not seem to have a core taxonomy, but rather shows how existing attacks can be implemented with RAMEN (their framework). Attack bounded by total number of queries.

2. Does not explicitly use training data, but the data used for training local models is labeld by one of the target models (VirusTotal)
    * [Toward an Effective Black-Box Adversarial Attack on Functional JavaScript Malware against Commercial Anti-Virus](https://dl.acm.org/doi/pdf/10.1145/3459637.3481956) , No Code: `loss function` - Attacks against Javascript Malware detection systems. Full black-box scenario with unlimited query access. Train substitute model (using term frequency as features), with a PGD + seq2seq attack on the proxy model. PGD is launched on the term frequencies, followed by building adversarial vocabulary. Injection code is sampled (for the best variant) using a seq2seq model.
