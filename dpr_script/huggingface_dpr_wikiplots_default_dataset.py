#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import logging
import os

import numpy

_DUMMY_DATASET_SIZE = 10000

try:
    import datasets
except ImportError:
    print("Please run `pip install datasets`")

try:
    import more_itertools
except ImportError:
    print("Please run `pip install more_itertools`")

try:
    from jsonlines import jsonlines
except ImportError:
    print("Please run `pip install jsonlines`")

try:
    from blingfire import text_to_sentences
except ImportError:
    print("Please run `pip install blingfire`")

try:
    import faiss
except ImportError:
    print(
        "Please see Faiss installation instructions at - https://github.com/facebookresearch/faiss/blob/master/INSTALL.md")

_HOMEPAGE = "https://github.com/facebookresearch/DPR"

_CITATION = """
@article{Karpukhin2020DensePR,
  title={Dense Passage Retrieval for Open-Domain Question Answering},
  author={V. Karpukhin and Barlas Oğuz and Sewon Min and Patrick Lewis and Ledell Yu Wu and Sergey Edunov and Danqi Chen and Wen-tau Yih},
  journal={ArXiv},
  year={2020},
  volume={abs/2004.04906}
}"""

_DESCRIPTION = """\
 Datasets script for reading datasets prepared for DPR and adding a Faiss index.
 The format for all the datasets read here is a jsonl file with a separate numpy vector file saved in npz format.
 The code for this is adapted from the https://huggingface.co/datasets/wiki_dpr dataset.
"""

_VERSION = datasets.Version("1.0.0")

_DESCRIPTION = """\
 The WritingPrompts dataset is over 300K short stories collected from the reddit forum /r/WritingPrompts/ .
 Each story is a creative writing exercise following a prompt.
"""


class StoryDprWikiDefaultDatasetConfig(datasets.BuilderConfig):
    """ BuilderConfig for DPR Datasets"""

    def __init__(self,
                 name="wikiplots_dpr_window_4_step_2_embeddings_index_compressed",
                 dataset_name="wikiplots",
                 data_url="https://github.com/dwlmt/story-datasets/raw/main/WikiPlots/wikiplots_20200701_dpr_window_4_step_2.jsonl.gz",
                 data_num_bytes=130341879,
                 data_checksum="7c0e119679473e609585ca8019b2d4add0b0184fe74a74e87b3ccfb1967dc771",
                 vectors_url="https://drive.google.com/uc?export=download&id=10g8zH0mOBLS_tN2TD9thrl2RLRpkONjh",
                 vectors_num_bytes=2996591652,
                 vectors_checksum="8968c448ae73a9432b9c015cfd500047b710c0996081b25efe1424189c1fc1b0",
                 description="Wikiplots dataset with DPR vectors from the multiset configuration.",
                 with_embeddings=True,
                 with_index=True,
                 index_name: str = "compressed",
                 embedding_dim: int = 768,
                 train_size: int = 250000,
                 index_worlds: int = 32,
                 index_ncentroids=4096,
                 index_code_size=64,
                 dummy: bool = False,
                 version=_VERSION,
                 **kwargs):

        self.dataset_name = dataset_name
        self.data_url = data_url
        self.vectors_url = vectors_url
        self.with_index = with_index
        self.index_name = index_name
        self.with_embeddings = with_embeddings
        self.embedding_dim = embedding_dim
        self.data_num_bytes = data_num_bytes
        self.data_checksum = data_checksum
        self.vectors_num_bytes = vectors_num_bytes
        self.vectors_checksum = vectors_checksum
        self.dummy = dummy
        self.train_size = train_size
        self.index_worlds = index_worlds
        self.index_code_size = index_code_size
        self.index_ncentroids = index_ncentroids

        self.index_file = f"{name}-{index_name}.faiss"

        if self.dummy:
            self.index_file = "dummy." + self.index_file

        if self.dummy:
            self.index_file = "dummy." + self.index_file

        super(StoryDprWikiDefaultDatasetConfig, self).__init__(name=name,
                                                               description=description,
                                                               version=version,
                                                               **kwargs)


class StoryWikiDefaultDprDataset(datasets.GeneratorBasedBuilder):
    """
     Default Wiki version of the dataset with the compressed index.
    """
    BUILDER_CONFIG_CLASS = StoryDprWikiDefaultDatasetConfig

    def _info(self):

        download_checksums = {self.config.data_url: {"num_bytes": self.config.data_num_bytes,
                                                     "checksum": self.config.data_checksum}}

        if self.config.vectors_url is not None:
            download_checksums[self.config.vectors_url] = {"num_bytes": self.config.vectors_num_bytes,
                                                           "checksum": self.config.vectors_checksum}

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "embeddings": datasets.Sequence(datasets.Value("float32")),
                }
            )
            if self.config.with_embeddings
            else datasets.Features(
                {"id": datasets.Value("string"), "text": datasets.Value("string"), "title": datasets.Value("string")}
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            #download_checksums=download_checksums
        )

    '''
    def _post_processing_resources(self, split):
        if self.config.with_index:
            return {"embeddings_index": self.config.index_file}
        else:
            return {}
    '''

    def _split_generators(self, dl_manager):
        files_to_download = {"data_file": self.config.data_url}
        downloaded_files = dl_manager.download_and_extract(files_to_download)
        if self.config.with_embeddings:
            downloaded_files["vectors_files"] = dl_manager.download([self.config.vectors_url])
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=downloaded_files),
        ]

    def _generate_examples(self, data_file, vectors_files=None):
        vec_idx = 0
        vecs = []

        with jsonlines.open(data_file, mode='r') as reader:

            for i, obj in enumerate(reader):

                # For dummy only load the first 10000.
                if self.config.dummy and i == _DUMMY_DATASET_SIZE:
                    break

                id = obj['id']
                title = obj['title']
                text = obj['text']

                if self.config.with_embeddings:
                    if vec_idx >= len(vecs):

                        if len(vectors_files) == 0:
                            logging.warning("Ran out of vector files at index {}".format(i))
                            break

                        # Currently all the files are assumed to have just one array.
                        vecs = numpy.load(vectors_files.pop(0))['arr_0']
                        # Reshape if the array has been expanded to a dim of 3.
                        if len(vecs.shape) == 3:
                            vecs = numpy.squeeze(vecs, axis=0)
                      
                        vec_idx = 0
                        vec = vecs[vec_idx]
                    yield id, {"id": id, "text": text, "title": title, "embeddings": vec}
                    vec_idx += 1
                else:
                    yield id, {
                        "id": id,
                        "text": text,
                        "title": title,
                    }

    def _post_process(self, dataset, resources_paths):
        if self.config.with_index:

            index_file = None#resources_paths["embeddings_index"]

            if index_file is not None and os.path.exists(index_file):
                dataset.load_faiss_index("embeddings", index_file)
            else:

                if "embeddings" not in dataset.column_names:
                    raise ValueError("Couldn't build the index because there are no embeddings.")
                import faiss

                logging.info(f"Building {self.config.name} faiss index")
                if self.config.index_name == "exact":
                    d = self.config.embedding_dim
                    index = faiss.IndexHNSWFlat(d, self.config.index_worlds, faiss.METRIC_INNER_PRODUCT)
                    dataset.add_faiss_index("embeddings", custom_index=index)
                else:
                    d = self.config.embedding_dim
                    quantizer = faiss.IndexHNSWFlat(d, self.config.index_worlds, faiss.METRIC_INNER_PRODUCT)
                    ivf_index = faiss.IndexIVFPQ(quantizer, d, self.config.index_ncentroids,
                                                 self.config.index_code_size, 8,
                                                 faiss.METRIC_INNER_PRODUCT)
                    ivf_index.own_fields = True
                    quantizer.this.disown()
                    dataset.add_faiss_index(
                        "embeddings",
                        custom_index=ivf_index,
                        train_size=self.config.train_size,
                        faiss_verbose=True
                    )

                logging.info(f"Saving {self.name} faiss index")
                #dataset.save_faiss_index("embeddings", index_file)

        return dataset
