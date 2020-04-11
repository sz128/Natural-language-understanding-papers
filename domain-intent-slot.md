
# NLU papers for domain-intent-slot
A list of recent papers regarding natural language understanding and spoken language understanding. <br>
It contains sequence labelling, sentence classification, dialogue act classification, dialogue state tracking and so on.

 * A review about NLU datasets for task-oriented dialogue is [here](https://github.com/sz128/NLU_datasets_for_task_oriented_dialogue).
 * There is an [implementation](https://github.com/sz128/slot_filling_and_intent_detection_of_SLU) of joint training of slot filling and intent detection for NLU, which is evaluated on ATIS, SNIPS, the Facebook’s multilingual dataset, MIT corpus, E-commerce Shopping Assistant (ECSA) dataset and CoNLL2003 NER datasets.

# Bookmarks
  * [Variant networks for different semantic representations](#1-variant-networks-for-different-semantic-representations)
  * [Robustness to ASR-error](#2-robustness-to-ASR-error)
  * [Zero-shot learning and domain adaptation](#3-zero-shot-learning-and-domain-adaptation)

## 1 Variant networks for different semantic representations
### 1.1 Domain-intent-slot
  * [Using Recurrent Neural Networks for Slot Filling in Spoken Language Understanding](https://ieeexplore.ieee.org/document/6998838/). Grégoire Mesnil, et al.. TASLP, 2015. [[Code+data](https://github.com/mesnilgr/is13)]
  * [Attention-based recurrent neural network models for joint intent detection and slot filling](https://pdfs.semanticscholar.org/84a9/bc5294dded8d597c9d1c958fe21e4614ff8f.pdf). Bing Liu and Ian Lane. InterSpeech, 2016. [[Code1](https://github.com/HadoopIt/rnn-nlu)] [[Code2](https://github.com/applenob/RNN-for-Joint-NLU)]
  * [Encoder-decoder with Focus-mechanism for Sequence Labelling Based Spoken Language Understanding](https://speechlab.sjtu.edu.cn/papers/sz128-zhu-icassp17.pdf). Su Zhu and Kai Yu. ICASSP, 2017. [[Code](https://github.com/sz128/SLU_focus_and_crf)]
  * [Neural Models for Sequence Chunking](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14776/14262). Fei Zhai, et al. AAAI, 2017.
  * [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354). Xuezhe Ma, Eduard Hovy. ACL, 2016.
  * [A Bi-model based RNN Semantic Frame Parsing Model for Intent Detection and Slot Filling](http://aclweb.org/anthology/N18-2050). Yu Wang, et al. NAACL 2018.
  * [A Self-Attentive Model with Gate Mechanism for Spoken Language Understanding](http://aclweb.org/anthology/D18-1417). Changliang Li, et al. EMNLP 2018. [from Kingsoft AI Lab]
  * [Joint Slot Filling and Intent Detection via Capsule Neural Networks](https://arxiv.org/pdf/1812.09471.pdf). Chenwei Zhang, et al. 2018. 
  * [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/pdf/1902.10909.pdf). Qian Chen, et al.. Arxiv 2019.
  * [A Novel Bi-directional Interrelated Model for Joint Intent Detection and Slot Filling](https://www.aclweb.org/anthology/P19-1544.pdf). Haihong E, et al. ACL, 2019.
  * [A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding](https://www.aclweb.org/anthology/D19-1214.pdf). Libo Qin, et al. EMNLP-IJCNLP, 2019.

### 1.2 Dialogue act (act-slot-value triples)
  * [Improving Slot Filling in Spoken Language Understanding with Joint Pointer and Attention](http://aclweb.org/anthology/P18-2068). Lin Zhao and Zhe Feng. ACL, 2018.
  * [A Hierarchical Decoding Model for Spoken Language Understanding from Unaligned Data](https://arxiv.org/abs/1904.04498). Zijian Zhao, et al. ICASSP 2019. [SJTU]

### 1.3 Hierarchical Representations
  * [Semantic Parsing for Task Oriented Dialog using Hierarchical Representations](http://aclweb.org/anthology/D18-1300). Sonal Gupta, et al. EMNLP 2018. [from Facebook AI Research]
  
## 2 Robustness to ASR-error
 * [Discriminative spoken language understanding using word confusion networks](http://www.matthen.com/assets/pdf/Discriminative_Spoken_Language_Understanding_Using_Word_Confusion_Networks.pdf). Matthew Henderson, et al.. SLT, 2012. [[Data](https://www.repository.cam.ac.uk/handle/1810/248271;jsessionid=D40F449AE8CD5D93EF215715D1726E13)]
 * [Using word confusion networks for slot filling in spoken language understanding](http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1353.pdf). Xiaohao Yang and Jia Liu. Interspeech, 2015.
 * [Joint Online Spoken Language Understanding and Language Modeling with Recurrent Neural Networks](http://www.aclweb.org/anthology/W16-3603). Bing Liu and Ian Lane. SIGDIAL, 2016. [[Code](https://github.com/HadoopIt/joint-slu-lm)]
 * [Robust Spoken Language Understanding with unsupervised ASR-error adaptation](https://speechlab.sjtu.edu.cn/papers/sz128-zhu-icassp18.pdf). Su Zhu, et al.. ICASSP, 2018.
 * [Neural Confnet Classification: Fully Neural Network Based Spoken Utterance Classification Using Word Confusion Networks](http://mirlab.org/conference_papers/International_Conference/ICASSP%202018/pdfs/0006039.pdf). Ryo Masumura, et al.. ICASSP, 2018.
 * [From Audio to Semantics: Approaches to end-to-end spoken language understanding](https://arxiv.org/abs/1809.09190). Parisa Haghani, et al.. SLT, 2018. [Google]
 * [Robust Spoken Language Understanding with Acoustic and Domain Knowledge](https://dl.acm.org/doi/10.1145/3340555.3356100). Hao Li, et al.. ICMI, 2019. [SJTU]
 * [Adapting Pretrained Transformer to Lattices for Spoken Language Understanding](https://www.csie.ntu.edu.tw/~yvchen/doc/ASRU19_LatticeSLU.pdf). Chao-Wei Huang and Yun-Nung Chen. ASRU, 2019. [[Code](https://github.com/MiuLab/Lattice-SLU)]
 * [Learning ASR-Robust Contextualized Embeddings for Spoken Language Understanding](https://www.csie.ntu.edu.tw/~yvchen/doc/ICASSP20_SpokenVec.pdf). Chao-Wei Huang and Yun-Nung Chen. ICASSP, 2020.
 
 ## 3 Zero-shot learning and domain adaptation
 ### 3.1 Zero-shot learning
  * [A model of zero-shot learning of spoken language understanding](http://www.anthology.aclweb.org/D/D15/D15-1027.pdf). Majid Yazdani and James Henderson. EMNLP, 2015.
  * [Zero-shot Learning Of Intent Embeddings For Expansion By Convolutional Deep Structured Semantic Models](https://www.csie.ntu.edu.tw/~yvchen/doc/ICASSP16_ZeroShot.pdf). Yun-Nung Chen, et al.. ICASSP 2016.
  * [Online Adaptative Zero-shot Learning Spoken Language Understanding Using Word-embedding](https://ieeexplore.ieee.org/document/7178987/).  Emmanuel Ferreira, et al. ICASSP 2015.
  * [Label Embedding for Zero-shot Fine-grained Named Entity Typing](https://sentic.net/label-embedding-for-zero-shot-named-entity-typing.pdf). Yukun Ma et al. COLING, 2016.
  * [Towards Zero-Shot Frame Semantic Parsing for Domain Scaling](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0518.PDF).  Ankur Bapna, et al. Interspeech, 2017.
  * [Concept Transfer Learning for Adaptive Language Understanding](http://aclweb.org/anthology/W18-5047). Su Zhu and Kai Yu. SIGDIAL, 2018.
  * [An End-to-end Approach for Handling Unknown Slot Values in Dialogue State Tracking](http://aclweb.org/anthology/P18-1134). Puyang Xu and Qi Hu. ACL, 2018.
  * [Large-Scale Multi-Domain Belief Tracking with Knowledge Sharing](http://aclweb.org/anthology/P18-2069). Osman Ramadan, et al.. ACL, 2018. [[Data](http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/)]
  * [Zero-Shot Adaptive Transfer for Conversational Language Understanding](https://arxiv.org/abs/1808.10059). Sungjin Lee, et al.. Arxiv 2018. [Microsoft]
  * [Robust Zero-Shot Cross-Domain Slot Filling with Example Values](https://www.aclweb.org/anthology/P19-1547.pdf). Darsh J Shah, et al.. ACL, 2019.
 ### 3.2 Few-shot learning
  * [Few-shot classification in Named Entity Recognition Task](https://arxiv.org/pdf/1812.06158.pdf). Alexander Fritzler, et al. SAC, 2019.
  * [Few-Shot Text Classification with Induction Network](https://arxiv.org/pdf/1902.10482.pdf). Ruiying Geng, et al. Arxiv 2019.
  * [Few-Shot Sequence Labeling with Label Dependency Transfer and Pair-wise Embedding](https://arxiv.org/abs/1906.08711). Yutai Hou, et al.. Arxiv 2019.
 ### 3.3 Domain adaptation
  * [Domain Attention with an Ensemble of Experts](http://www.karlstratos.com/publications/acl17ensemble.pdf). Young-Bum Kim, et al.. ACL, 2017.
  * [Adversarial Adaptation of Synthetic or Stale Data](http://karlstratos.com/publications/acl17adversarial.pdf). Young-Bum Kim, et al.. ACL, 2017.
  * [Fast and Scalable Expansion of Natural Language Understanding Functionality for Intelligent Agents](http://aclweb.org/anthology/N18-3018). Anuj Goyal, et al. NAACL, 2018. [from Amazon Alexa Machine Learning]
  * [Bag of Experts Architectures for Model Reuse in Conversational Language Understanding](http://aclweb.org/anthology/N18-3019). Rahul Jha, et al.. NAACL, 2018. [from Microsoft Corporation]
  * [Data Augmentation with Atomic Templates for Spoken Language Understanding](https://www.aclweb.org/anthology/D19-1375.pdf). Zijian Zhao, et al. EMNLP-IJCNLP, 2019. [SJTU]
  * [Prior Knowledge Driven Label Embedding for Slot Filling in Natural Language Understanding](https://arxiv.org/abs/2003.09831). Su Zhu, et al.. TASLP, 2020. [SJTU]
 
