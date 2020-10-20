#!/usr/bin/env python3
"""Huggingface datasets script file for processing  """
from __future__ import absolute_import, division, print_function

import os

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

_CITATION = ""

_DESCRIPTION = """\
 English language plots taken from the English Wikipedia from films, books, plays and other narrative forms. The dataset
 has 132,358 plots in total.
"""

_VERSION = datasets.Version("1.0.0")

_URL = "https://github.com/dwlmt/story-datasets/raw/main/WikiPlots/wikiplots_20200701.jsonl.gz"

_HOMEPAGE = "https://github.com/markriedl/WikiPlots"


class WikiPlotsDatasetConfig(datasets.BuilderConfig):
    """ BuilderConfig for WikiPlots"""

    def __init__(self, sentences_per_passage_num: int = 1, sentences_per_passage_slide: int = 1, **kwargs):
        """
        Args:
            sentences_per_passage_num (int): How many sentences should be in a single block. Default = 1.
            **kwargs: keyword arguments forwarded to super.
        """
        self.sentences_per_passage_num = sentences_per_passage_num
        super(WikiPlotsDatasetConfig, self).__init__(**kwargs)


class WikiPlotsDatasetConfig(datasets.GeneratorBasedBuilder):
    """English language plots taken from the English Wikipedia from films, books, plays and other narrative forms. The dataset
 has 132,358 plots in total.

    An example of usage:

    from datasets import load_dataset
    dataset = load_dataset('{THIS SCRIPT URL}/hugging_face_dataset.py',name='wikiplots_sentence', split='train')

    """

    BUILDER_CONFIG_CLASS = WikiPlotsDatasetConfig
    BUILDER_CONFIGS = [
        WikiPlotsDatasetConfig(name="wikiplots_sentence", description="Wikiplots split by sentence.",
                                    version=_VERSION),
        WikiPlotsDatasetConfig(name="wikiplots_passage",
                                    description="Wikiplots split by passages of 4 sentences.",
                                    sentences_per_passage_num=4, version=_VERSION)
    ]

    def _info(self):

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),  # The Wikipedia page id.
                    "url": datasets.Value("string"),  # Link back to the original Wikipedia page.
                    "title": datasets.Value("string"),  # Title of the work.
                    "passages": [  # Split text.
                        {"text": datasets.Value("string"),  # The text of the passage.
                         "id": datasets.Value("string"),  # Format 'id-seq_num'
                         "seq_num": datasets.Value("int32")}  # Ordered identifier for the passage in the story.
                    ],
                }
            ),
            download_checksums={_URL: {"num_bytes": 109300457,
                                       "checksum": "7fe76225dcff4ff53830f7272d298a9c2f57e091f76411c652db7b2fed04ed78"}},
            supervised_keys=None,
            version=_VERSION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Train only as will use datasets functionality to split dyanamically."""

        dl_file = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": dl_file,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        """ Yields an example for each story with text nested in passages.
        """

        with jsonlines.open(filepath, mode='r') as reader:

            for obj in reader:

                passages = []

                sentences = text_to_sentences(f"{obj['text']}").split('\n')
                passages_text = list(more_itertools.chunked(sentences, self.config.sentences_per_passage_num))

                for i, p in enumerate(passages_text):
                    passages.append({"text": " ".join(p), "seq_num": int(i), "id": f"{obj['id']}-{i}"})

                story_dict = {"id": str(obj["id"]), "title": obj["title"], "url": obj["url"], "passages": passages}

                yield str(story_dict["id"]), story_dict
