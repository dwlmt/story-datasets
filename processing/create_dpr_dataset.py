from typing import List, OrderedDict

import fire
import more_itertools
import numpy
import torch
from jsonlines import jsonlines
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

try:
    from blingfire import text_to_sentences
except ImportError:
    print("Please run `pip install blingfire`")


class ProcessDPRDataset(object):

    def create(self, datasets: List[str], base_output_file, window_size: int = 4, window_step=2, max_length=512,
               batch_size=8,
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

        if torch.cuda.is_available():
            ctx_encoder = ctx_encoder.cuda()

        with jsonlines.open(f'{base_output_file}.jsonl', mode='w') as writer:
            writer.write(output_json_list)

        vectors = []
        with torch.no_grad():

            def encode_vector_batch(batch_list):

                tokens = ctx_tokenizer(batch_list, truncation=True, padding="longest", max_length=max_length,
                                       return_tensors="pt")

                if torch.cuda.is_available():

                    for key in tokens.keys():
                        tokens[key] = tokens[key].cuda()

                vector = ctx_encoder(
                    **tokens)[0]

                if len(vector.size()) == 1:
                    vector = vector.unsqueeze(dim=0)

                vector = vector.cpu().numpy()

                return vector

            print(f"Encoding vectors, total {len(output_json_list)}")

            batch_list = []
            examples = tqdm(output_json_list)
            for example in examples:

                batch_list.append(example['text'])

                if len(batch_list) == batch_size:
                    # examples.set_description(f"Encoding batch: {batch_list}")
                    vectors.append(encode_vector_batch(batch_list))

                    batch_list = []

            if len(batch_list) > 0:
                vectors.append(encode_vector_batch(batch_list))

        vectors = numpy.concatenate(vectors, axis=0)

        print("Vector shape: ", vectors.shape)

        numpy.savez_compressed(f'{base_output_file}.npz', [vectors])


if __name__ == '__main__':
    fire.Fire(ProcessDPRDataset)
