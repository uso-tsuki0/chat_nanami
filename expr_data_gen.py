
import os
from pathlib import Path
from parser import BaseParser
import json
import tqdm


class ExprDataGenerator():
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.parser = BaseParser()

    def generate(self, chara_name):
        output_file = os.path.join(self.output_folder, f'{chara_name}.jsonl')
        directory_path = Path(self.input_folder)
        with open(output_file, 'w', encoding='utf-8') as f:
            for file_path in tqdm.tqdm(list(directory_path.glob('*.json'))):
                parsed_maps = self.parser.merge_text(self.parser.parse(file_path))
                for map in tqdm.tqdm(parsed_maps):
                    character = map['character']
                    face = map['face']
                    text_cn = map['text_cn']
                    result = {
                        'character': character,
                        'face': face,
                        'text_cn': text_cn
                    }
                    if character == chara_name:
                        line = json.dumps(result, ensure_ascii=False)
                        f.write(line + '\n')


if __name__ ==  "__main__":
    generator = ExprDataGenerator('data/jsons_full', 'data/expr_data')
    generator.generate('七海')
        