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

_CITATION = """\
@inproceedings{fan-etal-2018-hierarchical,
    title = "Hierarchical Neural Story Generation",
    author = "Fan, Angela  and
      Lewis, Mike  and
      Dauphin, Yann",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P18-1082",
    doi = "10.18653/v1/P18-1082",
    pages = "889--898",
}
"""

_DESCRIPTION = """\
 The WritingPrompts dataset is over 300K short stories collected from the reddit forum /r/WritingPrompts/ .
 Each story is a creative writing exercise following a prompt.
"""

_VERSION = "1.0.0"

_URL = "https://github.com/dwlmt/story-datasets/raw/main/WritingPrompts/WritingPrompts.tar.gz"

_HOMEPAGE = "https://research.fb.com/publications/hierarchical-neural-story-generation/"


class WritingPromptsDatasetConfig(datasets.BuilderConfig):
    """ BuilderConfig for WritingPrompts"""

    def __init__(self, sentences_per_passage_num: int = 1, **kwargs):
        """
        Args:
            sentences_per_passage_num (int): How many sentences should be in a single block.
            **kwargs: keyword arguments forwarded to super.
        """
        self.sentences_per_passage_num = sentences_per_passage_num
        super(WritingPromptsDatasetConfig, self).__init__(**kwargs)


class WritingPromptsDataset(datasets.GeneratorBasedBuilder):
    """The WritingPrompts dataset is over 300K short stories collected from the reddit forum /r/WritingPrompts/ .
        Each story is a creative writing exercise following a prompt.

    An example of usage:

    from datasets import load_dataset
    dataset = load_dataset('{THIS SCRIPT URL}/hugging_face_dataset.py',name='writing_prompts_sentence', split='test')

    """

    BUILDER_CONFIG_CLASS = WritingPromptsDatasetConfig
    BUILDER_CONFIGS = [
        WritingPromptsDatasetConfig(name="writing_prompts_sentence", description="Writing Prompts split by sentence."),
        WritingPromptsDatasetConfig(name="writing_prompts_passage",
                                    description="Writing Prompts split by passages of 4 sentences.",
                                    sentences_per_passage_num=4)
    ]

    def _info(self):

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),  # The line number from source file.
                    "title": datasets.Value("string"),  # The prompt.
                    "passages": [  # Split text.
                        {"text": datasets.Value("string"),  # The text of the passage.
                         "id": datasets.Value("string"),  # Format 'id-seq_num'
                         "seq_num": datasets.Value("int32")}  # Ordered identifier for the passage in the story.
                    ],
                }
            ),
            download_checksums={_URL: {"num_bytes": 377596362,
                                       "checksum": "9ccbf6bc6d3e873185a8b358c8a5720a5bf045e001fc01abf6e92e73cd9f28f3"}},
            supervised_keys=None,
            version=_VERSION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns splits from train,valid,test.jsonl """

        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "WritingPrompts")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "test.jsonl"), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "valid.jsonl"),
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields an example for each story split by stories.
            The prompt is the title but also prepended to the main text.
        """

        with jsonlines.open(filepath, mode='r') as reader:

            for obj in reader:

                passages = []

                sentences = text_to_sentences(f"{obj['title']} {obj['text']}").split('\n')
                passages_text = list(more_itertools.chunked(sentences, self.config.sentences_per_passage_num))

                for i, p in enumerate(passages_text):
                    passages.append({"text": " ".join(p), "seq_num": int(i), "id": f"{obj['id']}-{i}"})

                story_dict = {"id": str(obj["id"]), "title": obj["title"], "passages": passages}

                yield str(story_dict["id"]), story_dict
