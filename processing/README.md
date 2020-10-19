# Processing scripts.

## create_dpr_dataset.py

Converts a dataset into a Huggingface dataset with vectors that can be used by RAG.

Uses DPR vectors defined in the following article:

@article{karpukhin2020dense,
title={Dense Passage Retrieval for Open-Domain Question Answering},
author={Karpukhin, Vladimir and O{\u{g}}uz, Barlas and Min, Sewon and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau},
journal={arXiv preprint arXiv:2004.04906},
year={2020}
}

DPR is used as this is the information retrieval component for RAG

@article{Lewis2020RetrievalAugmentedGF,
title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
author={Patrick Lewis and E. Perez and Aleksandara Piktus and F. Petroni and V. Karpukhin and Naman Goyal and Heinrich Kuttler and M. Lewis and W. Yih and Tim Rockt{\"a}schel and S. Riedel and Douwe Kiela},
journal={ArXiv},
year={2020},
volume={abs/2005.11401}
}

The input expected is jsonlines format with a text field required per line.

