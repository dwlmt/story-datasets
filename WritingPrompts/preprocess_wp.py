#!/usr/bin/env python3
import os
from collections import OrderedDict

try:
    import fire
except ImportError:
    print("Please run `pip install fire`")

try:
    import more_itertools
except ImportError:
    print("Please run `pip install more_itertools`")

try:
    from jsonlines import jsonlines
except ImportError:
    print("Please run `pip install jsonlines`")

def ensure_dir(file_path):
    """ Make sure the output path exists.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class WritingPrompts(object):
    """
        Convert the source and target prompts and story files into a simple jsonl format. The main reason is just
        to make it simpler to process from a single file.
    """

    def jsonl(self, prompts_file: str, stories_file: str, output_file: str):

        print(f"Write output to: {output_file}")
        ensure_dir(output_file)

        print(f"Read prompts from: {prompts_file}")
        print(f"Read stories from: {stories_file}")

        stories_list = []

        with open(prompts_file, encoding="utf8") as prompts, open(stories_file, encoding="utf8") as stories:
            for i, (prompt, story) in enumerate(zip(prompts, stories)):
                prompt = prompt.replace("<newline>", "\n")
                story = story.replace("<newline>", "\n")

                story_dict = OrderedDict()
                story_dict["id"] = i
                story_dict["title"] = prompt
                story_dict["text"] = story

                stories_list.append(story_dict)

        with jsonlines.open(output_file, mode='w') as writer:

            for story in stories_list:
                writer.write(story)


if __name__ == '__main__':
    fire.Fire(WritingPrompts)
