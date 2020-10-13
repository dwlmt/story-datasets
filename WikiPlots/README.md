# WikiPlots

## Source

WikiPlots is adapted from Mark Reidl scripts found at - https://github.com/markriedl/WikiPlots .

The main differences are that this extract is from a newer version of Wikipedia dump (20200701) and as such as more plots,
132,358 up from 112,936. It is also extracted is jsonllines format with the whole text being in a single
block rather than pre-split into sentences.

# Format

The main dataset is in the **wikiplots_20200701.jsonl.gz** file.

Per json line the format is:

- **id:** The Id of Wikipedia source page.
- **title:** The title of the source page.
- **text:** The extracted plot text.
- **url:** Link back to the original Wikipedia page.

Additionally the file has been shuffled so can be used for as is for dynamic train/validation/test splits.


## Huggingface Datasets

**./hugging_face_dataset.py** contains a [Huggingface datasets](https://github.com/huggingface/datasets) script to read stories with one story per training set
example and stories broken down into sentences or longer passages nested under "passages" in the example
dictionary. Please see dataset.features in the script for the format.

The sentences are split using [BlingFire](https://github.com/microsoft/BlingFire). Either single sentences or passages
grouped by 4 sentence blocks are available as configurations.


