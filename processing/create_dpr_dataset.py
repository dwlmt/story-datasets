from typing import List, OrderedDict

import fire
import more_itertools
import numpy
import torch
from jsonlines import jsonlines
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

try:
    from blingfire import text_to_sentences
except ImportError:
    print("Please run `pip install blingfire`")


class ProcessDPRDataset(object):

    def create(self, datasets: List[str], base_output_file, window_size: int = 4, window_step=2,
               rag_context_encoder="facebook/dpr-ctx_encoder-multiset-base"):

        if isinstance(datasets, str):
            datasets = [datasets]

        id = 0

        output_json_list = []
        for dataset in datasets:

            with jsonlines.open(dataset) as reader:
                for obj in reader:
                    # print(f"{obj['title']} - {obj['text']}")

                    sentences = text_to_sentences(f"{obj['text']}").split('\n')
                    passages = more_itertools.windowed(sentences, n=window_size, step=window_step, fillvalue=" ")
                    for i, p in enumerate(passages):
                        joined_text = " ".join(p).strip()

                        passage_dict = OrderedDict()
                        passage_dict["id"] = f"{id}"
                        passage_dict["title"] = f"{obj['id']}-{i}: {obj['title']}"
                        passage_dict["text"] = joined_text

                        print(f"Passage: {passage_dict}")
                        output_json_list.append(passage_dict)

                        id += 1

        ctx_encoder = DPRContextEncoder.from_pretrained(rag_context_encoder)
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

        with jsonlines.open(f'{base_output_file}.jsonl', mode='w') as writer:
            writer.write(output_json_list)

        vectors = []
        with torch.no_grad():

            for example in output_json_list:
                print(f"Create vector for text: {example['id']} - {example['text']}")
                vectors.append(ctx_encoder(**ctx_tokenizer(example["text"], return_tensors="pt"))[0][0].numpy())

        vectors = numpy.stack(vectors)

        print("Vector shape: ", vectors.shape)

        numpy.savez_compressed(f'{base_output_file}.npz', [vectors])


if __name__ == '__main__':
    fire.Fire(ProcessDPRDataset)
