---
permalink: /
title: "About Me"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a second-year master's student in iSchool at UIUC. My research interests are natural language processing and learning science. I'm currently a member of [BLENDER Lab](http://blender.cs.illinois.edu/) and working on relation extraction. 

I have worked from home for over 2 years (well before COVID-19), so I'm pretty good at it. If you need help working from home, chat with me, maybe I can enlighten you.

I like road trips and have ridden to 5 states the last summer. I'm color blind but I paint, sometimes. I love cats (but have none) and mechanical keyboards (and have 3)!

<!-- , advised by Professor Heng Ji. Prior to joining BLENDER at UIUC, I received my bachelor’s degree in computer science at CSUST, where I was supervised by Dr. Zeng. -->

Download my CV [here](https://windchimeran.github.io/files/cv.pdf)

## Research Stories

  <!-- ### Joint Extraction of Entities and Relations (JERE) -->

  Joint Extraction of Entities and Relations (JERE) task is to extract entity-relation triplets from the plain text, usually in a supervised setting, e.g., 
  > Obama graduated from Columbia University and Harvard Law School, and he was the president of the Harvard Law Review.
  ->
  > [(Obama, graduate_from, Columbia University), (Obama, graduate_from, Harvard Law School), (Obama, president_of, Harvard Law Review)]
  
  At first, we reproduced a machine-translation-like baseline, [CopyRE](https://www.aclweb.org/anthology/P18-1047.pdf), which "translated" the sentence to triplets via Seq2Seq. CopyRE found an entity by predicting its position in the original sentence, and a relation by predicting from a predefined set. 
  When reproducing the CopyRE on the NYT dataset, we noticed the model weiredly relied on a mask for entity extraction:
  - With mask: F1 scores is as expected.
  - Without mask: F1 scores is down to 0.
  
  We then dug into the codes and equations and found a linear-algebra mistake hidden behind the codes ... We solved it and created a new system called [CopyMTL](https://arxiv.org/pdf/1911.10438.pdf), which was accepted by **AAAI 2020**.

  We deployed CopyMTL to a large-scale JERE dataset, DuIE. However, CopyMTL got a very low score (CopyMTL = 40+, others = 70+). We dug deeper into the root by doing error analysis on the outputs, and found that the performance decreased while the number of triplets per sentence increased. After ruling out other possible explanations, we thought of the notorious exposure bias problem in machine translation, which may be the culprit altering the extraction results. If the length of output sequence can be reduced, the effect of exposure bias can be mitigated ... Finally we solved it by turning the sequence to an Unordered-Multi-Tree, and built a new system, [Seq2UMTree](https://arxiv.org/pdf/2009.07503.pdf). This paper was accpepted by **Findings of EMNLP 2020**.

  Seq2UMTree is not perfect. We find that there are still some errors caused by the shortage of relations and linguistic patterns in the training set. We are working on building a better system and would love to chat about it. If you are interested, maybe we can have Zoom coffee!



<!-- ## Paper and Manuscript

(\* refers to equal contribution) -->



<!-- - <u>Ranran Haoran Zhang</u>\*, Qianying Liu\*, Aysa Xuemo Fan, Heng Ji, Daojian Zeng, Fei Cheng, Daisuke Kawahara, Sadao Kurohashi, **Minimize Exposure Bias of Seq2Seq Models in Joint Entity and Relation Extraction**. EMNLP2020 Findings. Preprint [here](https://arxiv.org/pdf/2009.07503.pdf).

- Qingyun Wang, Manling Li, Xuan Wang, Nikolaus Parulian, Guangxing Han, Jiawei Ma, Jingxuan Tu, Ying Lin, <u>Ranran Haoran Zhang</u>, Weili Liu, Aabhas Chauhan, Yingjun Guan, Bangzheng Li, Ruisong Li, Xiangchen Song, Heng Ji, Jiawei Han, Shih-Fu Chang, James Pustejovsky, David Liem, Ahmed Elsayed, Martha Palmer, Jasmine Rah, Cynthia Schneider, Boyan Onyshkevych. **COVID-19 Literature Knowledge Graph Construction and Drug Repurposing Report Generation**. Preprint [here](https://arxiv.org/pdf/2007.00576.pdf).

- Daojian Zeng\*, <u>Ranran Haoran Zhang</u>\*, Qianying Liu, **CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations
with Multi-Task Learning**. AAAI, 2020. Retrieved from [here](https://arxiv.org/pdf/1911.10438.pdf). -->

## Education

**University of Illinois Urbana-Champaign**

*MS in Information Management, 2019 - present*

Advisor: Heng Ji

------


**Changsha University of Science and Technology**

*BS in Computer Science, 2014 - 2018*

Advisor: Daojian Zeng

## Contact

Email: haoranz6 [AT] illinois [DOT] edu
