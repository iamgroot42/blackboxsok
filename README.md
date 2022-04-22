# blackboxsok

Below, we list the candidate experiments to run in each domain and also the corresponding source code.

## Selection Criteria and Other Information

We only select attacks that: 1) reported state-of-the-art (SOTA) results in the paper with enough baselines, 2) peer-reviewed or have open source implementations available. Note that there can exist multiple SOTA attacks as those SOTA methods are not individually compared. In that case, we include all of them. We plan to report experiments for: 1) 1-2 standard models (undefended) and 1-2 robust models (can be obtained from robust bench: <https://robustbench.github.io>), 2) targeted (with different target classes) and untargeted settings, 2) different attack strenths. 

## Image Domain

Black-box attacks are heavily investigated in image domain and hence, we can summarize and syatemize existing knowledge and also gain new insights. Below are the list of experiments to run. The candidate attacks are selected to reflect the current state-of-the-art in each black-box attack scenario.

We need to setting up the unified testing framework (estimated time: **1-2 weeks**, but may be less than this). This framework should 1) ensure the target model can only be accessed with (simulated) API calls, 2) users can choose the specific auxiliary information (e.g., local models) available to the attacker and the framework will load the corresponding modules, 3) should be able to figure the attack details such as `norm, perturbation threshold, targeted/untargeted, robust/standard models, different types of target classes (e.g., most/least likely) for targeted attack`. We can also refer to <https://blackboxbench.github.io> for some guidance on designing the framework.   

### Hard-label, No Access, No-/Partial-/Full Auxiliary Information

1. (Estimated time: **2-3 weeks**) Changes to the inputs, gradient or loss function w.r.t. source models: All of these works assume the same or similar training data, but we can also check their performance on some different datasets, which covers `partial-/full-auxiliary information` categories. Currently, no work is done for `no-auxiliary` information category.
    * [On Success and Simplicity: A Second Look at Transferable Targeted Attacks](https://proceedings.neurips.cc/paper/2021/file/30d454f09b771b9f65e3eaf6e00fa7bd-Paper.pdf), [Code](https://github.com/ZhengyuZhao/Targeted-Tansfer): `loss function` - Published at NeurIPS 2021, uses logits loss instead of traditional cross entropy loss and uses larger iterations. Seems to help more with targeted attack, but not untargeted attacks. There is no need to individually test this method, instead, many follow-up works consider this as a way to boost the performance of their method (for example, `Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation`). Therefore, this loss function will be used as an alternative option for the loss design. 
    * [A Little Robustness Goes a Long Way: Leveraging Robust Features for Targeted Transfer Attacks](https://proceedings.neurips.cc/paper/2021/file/50f3f8c42b998a48057e9d33f4144b8b-Paper.pdf), [Partial Code](https://github.com/microsoft/robust-models-transfer): `source model type` - Published at NeurIPS 2021, partial code (some pretrained models) is provided, but should be very easy to implement because the remaining things are just completing the standard attacks. Main message from the paper is, slightly robust source models show better transferability to unknown target models, once the local attacks are using with the attack listed above. 
    * [Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation](https://openreview.net/forum?id=i7FNvHnPvPc) [No Code]: **No Response** `loss function` - Preprint, main messsage is that, instead of minimizing loss of single adversarial example, we should seek adversarial examples locating at the low-value and flat region of the loss landscape. Should still be feasible to implement on our own, as the main method is using min-max optimization iteratively and the attack method is explained quite clearly. 
    * [Staircase Sign Method for Boosting Adversarial Attacks](https://arxiv.org/abs/2104.09722), [Code](https://github.com/qilong-zhang/Staircase-sign-method): `gradient computation` - Preprint, main message is, either using gradient sign or full gradient value is suboptimal for transferability. We should use some intermediate gradients between sign and the exact value. 
    * [Boosting Adversarial Transferability through Enhanced Momentum](https://arxiv.org/pdf/2103.10609.pdf), [Code](https://github.com/JHL-HUST/EMI): published at BMVC 2021, `gradient computation` enhances the momentum further for better transferability. If the spatial momentum method is unwilling to share the code, this method will be the `go-to` one for better gradient computation. 
    * [Adaptive Image Transformations for Transfer-based Adversarial Attack](https://arxiv.org/abs/2111.13844), [No Code]: **Contacted** `Input` - Preprint, adaptive input transformation for better transferability that requires training a encoder-decoder network such that it can output the optimal combination of existing input transformation techniques. This cannot be achieved in the `No-Auxiliary` information setting, though. The author promised to share the code as soon as possible. If not shared, an alternative method is to use the [AutoMA](https://ieeexplore.ieee.org/abstract/document/9599534), [Code](https://github.com/HaojieYuan/autoAdv). (Another alternative baseline is [Improving the Transferability of Adversarial Samples with Adversarial Transformations](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Improving_the_Transferability_of_Adversarial_Samples_With_Adversarial_Transformations_CVPR_2021_paper.pdf), but the author refused to share the code (says still in the releasing process), so do not think this is worth considering.) Implementing on our own will be time-consuming.  
    * [Improving Adversarial Transferability with Spatial Momentum](https://arxiv.org/abs/2203.13479), [No Code]: **No Response** `Gradient Computation` - Preprint, spatial momentum method for boosted transferability. It should still be feasible to implement on our own because the major changes in the momentum part is to add different image transformations to stabalize the gradients. 
    * [~~Exploring Transferable and Robust Adversarial Perturbation Generation from the Perspective of Network Hierarchy~~](https://arxiv.org/abs/2108.07033), [No Code]: `loss function` - Preprint, the main message is, the intermediate layers are more helpful than the final output layer, which is somewhat similar to the resource intensive transfer attacks based on intermediate features. Also a good resource to discover feature space transferable attacks. This mehod will not be included because the neuron attribution attacks below is shown to perform better and contains the code. 
    * [Improving Adversarial Transferability via Neuron Attribution-Based Attacks](https://arxiv.org/abs/2204.00008), [Code](https://github.com/jpzhang1810/naa): Published at CVPR 2022, this is the state-of-the-art feature-level attack and is shown to outperform the baseline attacks that are based on output layer manipulation. The authors have not pushed the source code to the repo yet, but promised to release as soon as possible (it's a recently accepted paper).
2. (Estimated time: **2-3 weeks**) Requires additional model (binary classifier, generator, some pretrained models):
    * [Data-Free Adversarial Perturbations for Practical Black-Box Attacks](https://arxiv.org/pdf/2003.01295.pdf), [No Code]:  **No Response**, Published at ICCV 2021, requires some pretrained model on different dataset, which has no overlap with the data of target models (originally designed for `partial auxiliary information`). We can extend this to full information case by using pretrained models on the same dataset. However, the approach should still be feasible to implement on our own. 
    * [Data-free Universal Adversarial Perturbation and Black-box Attack](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Data-Free_Universal_Adversarial_Perturbation_and_Black-Box_Attack_ICCV_2021_paper.pdf), [Code](https://bit.ly/3y0ZTIC): **No Response**, Published at ICCV 2021 and should be a good baseline method for partial/ no auxliary information setting, but the authors have not pushed the source code to the repo yet and did not respond to my email.  
    * [Adversarial Attack across Datasets](https://arxiv.org/pdf/2110.07718.pdf), [No Code]: **No Response** Preprint, requires training a meta-learner based training on a different dataset withgout label space overlap with the data of target models. Extension to full-auxiliary information is straightforward. We may attempt to implement on our own, not sure how much time it will take to train meta-learners. We may exclude this method in the end. 
    * [Perturbing Across the Feature Hierarchy to Improve Standard and Strict Blackbox Attack Transferability](https://arxiv.org/pdf/2004.14861.pdf), [No Code]: Published at NeurIPS 2020, **Contacted**, but the author cannot share the code, , requires training binary classifier that learns the target class distribution of the intermediate layers of source network. For `partial-auxiliary information` setting, the training data have partial overlap or no overlap, but the label space overlaps. But building the auxiliary classifier seems to feasible to implement on our own, and same logic applies to the paper below also. 
    * [Can Targeted Adversarial Examples Transfer When the Source and Target Models Have No Label Space Overlap?](https://arxiv.org/pdf/2103.09916.pdf), [No Code]: Published at NeurIPS 2021 **Contacted**, but the author cannot share the code. The paper is of similar idea as the paper above. However, the considered setting is, there is even no label space overlap and aims to find the good proxy class (of the target class in mind) for the source models. 
    * [~~On Generating Transferable Targeted Perturbations~~](https://arxiv.org/abs/2103.14641), [Code](https://github.com/Muzammal-Naseer/TTP): Published at ICCV 2021, requires training a generator for target class feature distribution. This method is outperformed by the first attack of this list in the targeted setting and is also designed only for targeted setting. 

### Full-Score, Limited/Unlimited Access, No-/Partial-/Full Auxiliary Information

No/partial auxiliary information (Estimated time: **1 week**).

1. No auxiliary information:
    * [Square Attack: a query-efficient black-box adversarial attack via random search](https://arxiv.org/abs/1912.00049) [Code](https://github.com/max-andr/square-attack): Published at ECCV 2020, still the SOTA without additional auxiliary information.
    * [Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks](https://arxiv.org/abs/2006.12834), [Code](https://github.com/fra31/sparse-rs): Published at AAAI 2022, current SOTA for L_0 norm. Maybe worth mentioning in the paper that evaluation of L_0 norms are missing, but this norm is really important for practical applications. 
    * [BayesOpt Adversarial Attack](https://openreview.net/forum?id=Hkem-lrtvH), [Code](https://github.com/rubinxin/BayesOpt_Attack): Published at ICLR 2020, mainly designed for low quqery regime and need to check if it performs better than `square-attack`.
2. Partial auxiliary information:
    * [Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks](https://arxiv.org/abs/2111.01714), [Code](https://github.com/boschresearch/meta-rs): Published at NeurIPS 2021, meta-learns the search distribution of square-attack and improves its performance compared to manual tuning.
    * not many attacks for this cases, but can check any of the methods in full-score category can be used to derive some attacks in this setting.
3. Full auxiliary information (Estimated time: **1-2 weeks**):
    * Test the combination of best transfer attacks and best no auxiliary information optimization attacks above and see if we can get SOTA performance (i.e., use local AE from transfer attacks to boost vanilla optimization attacks).
    * [Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks](https://arxiv.org/abs/2111.01714), [Code](https://github.com/boschresearch/meta-rs): Published at NeurIPS 2021, meta-learns the search distribution of square-attack and improves its performance compared to manual tuning. Unfortunately, this approach only applies to the square-attack. 
    * [Diversity can be Transferred: Output Diversification for White- and Black-box Attacks](https://arxiv.org/abs/2003.06878), [Code](https://github.com/ermongroup/ODS): Published at NeurIPS 2020, uses diversified sampling distribution from source models to boost SimBA attack. 
    * [Simulating Unknown Target Models for Query-Efficient Black-box Attacks](https://arxiv.org/abs/2009.00960), [Code](https://github.com/machanic/SimulatorAttack): Published at CVPR 2021, simulates target model and trains better source models along the black-box query process. However, this approach is not suitable for hard-label setting. 
    * [Learning Black-Box Attackers with Transferable Priors and Query Feedback](https://arxiv.org/abs/2010.11742), [Code](https://github.com/TrustworthyDL/LeBA): Published at NeurIPS 2021, hybrid attack with tunable local models using higher order gradients. However, this approach is not suitable for hard-label setting. 
    * [QueryNet: Attack by Multi-Identity Surrogates](https://arxiv.org/abs/2105.15010), [Code](https://github.com/allenchen1998/querynet): Preprint, smarter way of utilizing multiple surrogate models to provide better starting points for black-box attacks and also tunes local models in the attack process. Should be applicable to hard-label attack setting, but the effectiveness is unclear. 
    * [Boosting Black-Box Attack with Partially Transferred Conditional Adversarial Distribution](https://arxiv.org/abs/2006.08538), [Code](https://github.com/Kira0096/CGATTACK): Published at CVPR 2022, the evaluations are quite comprehensive and the main idea is to leverage some energy models to learn the hidden subspace of adversarial examples. This method is not applicable to hard-label setting. The authors are also maintaining [BlackboxBench](https://blackboxbench.github.io). 
    * [~~Black-Box Adversarial Attack with Transferable Model-based Embedding~~](https://arxiv.org/abs/1911.07140), [Code](https://github.com/TransEmbedBA/TREMBA) Published at ICLR 2020: uses the latent space representation as the starting point and deloys NES attack subsequently (only applies to gradient estimation attacks). It is already outperformed by many other attacks and deciced not to include it anymore. 
    * [Query-efficient Meta Attack to Deep Neural Networks](https://openreview.net/forum?id=Skxd6gSYDS), [Code](https://github.com/dydjw9/MetaAttack_ICLR2020): Puboished at ICLR 2020, uses meta-learning to extract attacks from prior attacks, and then leverages the attack pattern in the actual black-box attacks. Might be applicable to hard-label setting, but the effectiveness is unknown when we fine-tune the meta-learner on the new task (as the estimated gradient is actually much less informative). 

### Hard-label, Limited/Unlimited access, No-/Partial-/Full Auxiliary Information
Estimated time for all experiments: **1 week**

1. No auxiliary information:
    * [RayS: A Ray Searching Method for Hard-label Adversarial Attack](https://arxiv.org/abs/2006.12792), [code](https://github.com/uclaml/RayS): Published at KDD 2020, SOTA attack with no auxiliary information for L_infty setting. We will need to adapt this attack to targeted setting also, the original paper only does for untargeted setting. Applicable to Hybrid attack.
    * [Query Efficient Decision Based Sparse Attacks Against Black-Box Deep Learning Models](https://arxiv.org/abs/2202.00091), [No Code]: **No Response** Published at ICLR 2022, reported SOTA in L_0 setting. Applicable to hybrid attack. I will directly talk to the authors at ICLR this year. 
    * [Simple and Efficient Hard Label Black-box Adversarial Attacks in Low Query Budget Regimes](https://arxiv.org/abs/2007.07210), [Code]( https://github.com/satyanshukla/bayes_attack): Bayesian optimization, suits better for limited access scenarios.
    * [~~Hard-label Manifolds: Unexpected Advantages of Query Efficiency for Finding On-manifold Adversarial Examples~~](https://arxiv.org/abs/2103.03325), [No Code]: Preprint, possible new insights (double check paper details), from UFL security group. Update: did not gain enough insight and will not include this method. 
2. Partial auxiliary information:
    * No existing work found, but can combine (in the form of hybrid attack) attacks in no auxiliary information with attacks in `No Access, Partial Auxiliary Information` category to propose new attacks. Easy to implement.
3. Full auxiliary information:
    * Combine listed attacks in No-Auxiliary information with techniques found in full-auxiliary information+No-access category.
    * The local model tuning strategy identified in `QueryNet: Attack by Multi-Identity Surrogates` can also be used for this setting with almost no modifications.
    * The local model tuning strategy identified in `Query-efficient Meta Attack to Deep Neural Networks` may also be used for this setting, but requires thinking on how we leverage the estimated gradients in hard-label setting. 
    * [Progressive-Scale Boundary Blackbox Attack via Projective Gradient Estimation](https://arxiv.org/abs/2106.06056), [Code](https://github.com/AI-secure/PSBA): Published at ICML 2021, requires training an autoencoder (latent space projection) on the same train data (applicable to hybrid attack).

### Top-K, Limited/Unlimited Access, No-/Partial-/Full Auxiliary Information
Estimated time for all experiments: (**< 1 week**)
1. No Auxiliary information:
    * [NES attack](https://arxiv.org/abs/1804.08598), [Code](https://github.com/labsix/limited-blackbox-attacks): The only paper that considers Top-k prediction settings. Their main idea for top-k case should also be applicable to other attacks proposed for the full-score case.
    * Adapt the [BayesOpt Adversarial Attack](https://openreview.net/forum?id=Hkem-lrtvH)([Code](https://github.com/rubinxin/BayesOpt_Attack)) using the the attack strategy in `NES` attack and check if it gives better performace, especially in limited access setting.
    * Adapt the [Square Attack: a query-efficient black-box adversarial attack via random search](https://arxiv.org/abs/1912.00049) [Code](https://github.com/max-andr/square-attack) using the idea in `NES`.
2. Partial auxiliary information: adapt the [Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks](https://arxiv.org/abs/2111.01714), [Code](https://github.com/boschresearch/meta-rs), search the distribution on a different dataset instead of on the original dataset of the target model. This experiments should be enough for this category.
3. Full auxiliary information: [Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks](https://arxiv.org/abs/2111.01714), [Code](https://github.com/boschresearch/meta-rs): the meta-training distribution is obtained on the exact same dataset of the target model. This experiments should be enough for this category.

## Malware Domain

### Hard-label, No Access, No-/Partial-/Full Auxiliary Information

1. Requires no access to training data or target model architecture
    * [Adversarial EXEmples: A Survey and Experimental Evaluation of Practical Attacks on Machine Learning for Windows Malware Detection](https://arxiv.org/pdf/2008.07125.pdf), [Code](https://github.com/pralab/secml_malware): `genetic algorithm` `transfer` - Analysis of Windows malware. Proposes 3 attacks that outperform existing ones in white-box (and use for transfer) and black-box scenarios. The black-box variant is the simplest & tries to minimize the loss with a surrogate model. The paper does not seem to have a core taxonomy, but rather shows how existing attacks can be implemented with RAMEN (their framework). Attack bounded by total number of queries.

    * [Evading API Call Sequence Based Malware Classifiers](https://link.springer.com/chapter/10.1007/978-3-030-41579-2_2), Code Shared by Authors: Additive modifications that are more probable to occur in benign samples. Calls to libraries are modified (pointers) such that adversarial code is launched (IAT hooking). The adversarial functions have similar API signatures. Adversarial training does seem to be highly effective (~0% attack effectiveness), though.

2. Does not eplicitly use training data, has no access to API until launching attack, only same training distributions.
    * [Evading Malware Classifiers via Monte Carlo Mutant Feature Discovery](https://arxiv.org/abs/2106.07860), [Code](https://github.com/UMBC-DREAM-Lab/montemutacon): A "gray-box" setup where features used are same for both victim and adversary. Adversary uses surrogate models trained on test split, victim uses larger model trained on train split. No API is exposed, except when launching the attack. Monte-Carlo Search Trees with implementation modifications, using 12 possible mutations.

    * [Generating End-to-End Adversarial Examples for Malware Classifiers Using Explainability](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9207168), No Code **Contacted**: Use explanations fron interpretable models to bypass multi-feature-type malwares. First identifies relevant features, and then modify them selectively. Considers transferability of explainability, "first to leverage explainability algorithms from the adversary side, to generate and facilitate adversarial attacks". Using substitute model, adversary can figure out "some features" used in training. Tried both 'same' and 'diff' train datasets, coming from the same dataset.

### Hard-label, Unlimited Access, No-/Partial-/Full Auxiliary Information

1. Does not explicitly use training data, but the data used for training local models is labeled by one of the target models (VirusTotal)
    * [Toward an Effective Black-Box Adversarial Attack on Functional JavaScript Malware against Commercial Anti-Virus](https://dl.acm.org/doi/pdf/10.1145/3459637.3481956) , No Code **Contacted**: `loss function` - Attacks against Javascript Malware detection systems. Full black-box scenario with unlimited query access. Train substitute model (using term frequency as features), with a PGD + seq2seq attack on the proxy model. PGD is launched on the term frequencies, followed by building adversarial vocabulary. Injection code is sampled (for the best variant) using a seq2seq model.

2. Requires access to training data/distribution
    * [Semantics aware adversarial malware examples generation for black-box attacks](https://www.sciencedirect.com/science/article/abs/pii/S1568494621004294), No Code **Contacted**: Uses word embeddings with a substitute sequential (RNN) model. A joint model where the substitute (for target) and generative model (for generating attacks) shares parameters for efficient training. Tried both 'same' and 'diff' train datasets, coming from the same dataset.

   * [Best-Effort Adversarial Approximation of Black-Box Malware Classifiers](https://arxiv.org/pdf/2006.15725.pdf), [Code](https://github.com/Abdullah-B/Best-Effort-Adversarial-Approximation-of-Black-Box-Malware-Classifiers) - `transfer' : Convert byte-sequences to “images” using Hilbert curves and other processing techniques Assumes only API access to the target (unaware of features used to train, etc.). Adversary uses pre-trained image-classification models and trains the last layer, and continues training until similarity (diff in accuracy on some disjoint data) stays above a certain threshold, also utilizing data augmentation (provably functionality preserving). Same datasets are used for generating “disjoint” splits.Results are on getting “close” to the target model, not actually launching any attacks. Conversion to images means we can potentially utilize a vast variety of image-based attacks (with restricted perturbations, perhaps?) and then transfer the attacks
