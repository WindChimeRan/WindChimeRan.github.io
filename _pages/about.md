---
permalink: /
title: "About me"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a second-year master's student at UIUC. My research interest are natural language processing and learning science. I'm currently a member of [BLENDER Lab](http://blender.cs.illinois.edu/) and working on relation extraction. I love mechanical keyboard!

<!-- , advised by Professor Heng Ji. Prior to joining BLENDER at UIUC, I received my bachelor’s degree in computer science at CSUST, where I was supervised by Dr. Zeng. -->

Download my CV [here](https://windchimeran.github.io/files/cv.pdf)

## Research Stories

- **Joint Extraction of Entities and Relations (JERE)**

    JERE task is to extract entity-relation triplets from the plain text, usually in a supervised setting, e.g., 
    > Obama graduated from Columbia Unversity and Harvard Law School, and he was the president of the Harvard Law Review.
    ->
    > [(Obama, graduate_from, Columbia Unversity), (Obama, graduate_from, Harvard Law School), (Obama, president_of, Harvard Law Review)]
    
    1. At first, we reproduce a machine-translation like baseline, [CopyRE](https://www.aclweb.org/anthology/P18-1047.pdf), which translates the sentence to triplets via Seq2Seq. CopyRE finds the entity by predicting the position in the original sentence, and the relation by predicting from a predefined set. 
    When reproducing the CopyRE on the NYT dataset, we found it weiredly relies on a mask for entity extraction:
    - with mask: F1 scores is as expected.
    - without mask: F1 scores is down to 0.
    Then we digged into the code and equations. We found there is a linear-algebra mistake hidding behind the code ... Finally we solved it and got an [AAAI 2020](https://arxiv.org/pdf/1911.10438.pdf). The new system is called CopyMTL.

    2. We deploied CopyMTL to a large-scale Chinese JERE dataset, DuIE. However, CopyMTL still got a very low score (CopyMTL=40+, others=70+). Then we digged into the errors of the outputs. We found the performance decreases a lot with the number of triplets per sentence increases. This reminds us that the notorious exposure bias problem in machine translation can be the culprit here, altering our extraction results. If the output triplet sequence is formulated to multiple shortest ones, the effect of exposure bias can be minimized ... Finally we solve  it and got an [EMNLP 2020 Findings](https://arxiv.org/pdf/2009.07503.pdf). The new system is called Sequence-to-Unordered-Multi-Tree (Seq2UMTree).

    3. Seq2UMTree is not perfect. We found that there are still some errors caused by the shortage of relations and linguistic patterns in the training set. The importances of relations are different for models, which relies on the frequency in the training set, while the importances are equal for human annotators, who learn relation annotation from the guideline, with several exemplars for each relation ... To be continue!



<!-- ## Paper and Manuscript

(\* refers to equal contribution) -->



<!-- - <u>Ranran Haoran Zhang</u>\*, Qianying Liu\*, Aysa Xuemo Fan, Heng Ji, Daojian Zeng, Fei Cheng, Daisuke Kawahara, Sadao Kurohashi, **Minimize Exposure Bias of Seq2Seq Models in Joint Entity and Relation Extraction**. EMNLP2020 Findings. Preprint [here](https://arxiv.org/pdf/2009.07503.pdf).

- Qingyun Wang, Manling Li, Xuan Wang, Nikolaus Parulian, Guangxing Han, Jiawei Ma, Jingxuan Tu, Ying Lin, <u>Ranran Haoran Zhang</u>, Weili Liu, Aabhas Chauhan, Yingjun Guan, Bangzheng Li, Ruisong Li, Xiangchen Song, Heng Ji, Jiawei Han, Shih-Fu Chang, James Pustejovsky, David Liem, Ahmed Elsayed, Martha Palmer, Jasmine Rah, Cynthia Schneider, Boyan Onyshkevych. **COVID-19 Literature Knowledge Graph Construction and Drug Repurposing Report Generation**. Preprint [here](https://arxiv.org/pdf/2007.00576.pdf).

- Daojian Zeng\*, <u>Ranran Haoran Zhang</u>\*, Qianying Liu, **CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations
with Multi-Task Learning**. AAAI, 2020. Retrieved from [here](https://arxiv.org/pdf/1911.10438.pdf). -->

## Education

**University of Illinois Urbana-Champaign**

MS in Information Management, 2019 - present

Advisor: Heng Ji

------


**Changsha University of Science and Technology**

BS in Computer Science, 2014 - 2018

Advisor: Daojian Zeng

## Contact

Email: haoranz6 [AT] illinois [DOT] edu
