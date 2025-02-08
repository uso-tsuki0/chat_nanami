import json
import os


class BaseParser:
    """
    Parser to parse dialogue data from game scene json.
    """
    def __init__(self):
        pass

    def find_tuple(self, data_list, key):
        for tuple in data_list:
            if isinstance(tuple, list) and tuple[0] == key:
                return tuple

    def parse_diaologue(self, map):
        diaologue_tuple = map[1]
        character = diaologue_tuple[2][0]
        if character is None:
            character = 'user throught'
        text_jp = diaologue_tuple[0][1]
        text_en = diaologue_tuple[1][1]
        text_cn = diaologue_tuple[2][1]
        is_dialogue = (text_cn[0]=='「' and text_cn[-1]=='」') or (text_cn[0]=='『' and text_cn[-1]=='』')
        if is_dialogue:
            text_cn = text_cn[1:-1]
            text_jp = text_jp[1:-1]
        return character, text_jp, text_en, text_cn, is_dialogue
    
    def parse_stage(self, map):
        stage_tuple = self.find_tuple(map[4]['data'], 'stage')
        show_mode = stage_tuple[2].get('showmode', None)
        if show_mode and show_mode != 0:
            return stage_tuple[2].get('redraw', {}).get('imageFile', {}).get('file', None)
        else:
            return None
        
    def parse_face(self, map):
        face_tuple = self.find_tuple(map[4]['data'], 'face')
        if not face_tuple:
            return None
        show_mode = face_tuple[2].get('showmode', None)
        if show_mode and show_mode != 0:
            return face_tuple[2].get('redraw', {}).get('imageFile', {}).get('options', {}).get('face', None)
        else:
            return None
        
    
    def parse(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = []
        for i, scene in enumerate(data["scenes"]):
            if 'texts' in scene:
                for j, map in enumerate(scene["texts"]):
                    try:
                        character, text_jp, text_en, text_cn, is_diaologe = self.parse_diaologue(map)
                        stage = self.parse_stage(map)
                        face = self.parse_face(map)
                        result.append({
                            'character': character,
                            'text_jp': text_jp,
                            'text_en': text_en,
                            'text_cn': text_cn,
                            'is_dialogue': is_diaologe,
                            'stage': stage,
                            'face': face
                        })
                    except Exception as e:
                        print(f'Error at scene {i} map {j}')
                        print(e)
        return result
    

    def merge_text(self, map_list):
        result = []
        text_cache = {
            'text_jp': '',
            'text_en': '',
            'text_cn': ''
        }
        curr_character = None
        curr_is_dialogue = None
        curr_face = None
        curr_stage = None
        for map in map_list:
            if map['character'] == curr_character and map['is_dialogue'] == curr_is_dialogue and map['face'] == curr_face and map['stage'] == curr_stage:
                if text_cache['text_jp'][-1] in ['。', '！', '？', '…']:
                    text_cache['text_jp'] += map['text_jp']
                else:
                    text_cache['text_jp'] += ('。' + map['text_jp'])
                text_cache['text_en'] += (' ' + map['text_en'])
                if text_cache['text_cn'][-1] in ['。', '！', '？', '…']:
                    text_cache['text_cn'] += map['text_cn']
                else:
                    text_cache['text_cn'] += ('。' + map['text_cn'])
            else:
                if curr_character:
                    result.append({
                        'character': curr_character,
                        'text_jp': text_cache['text_jp'],
                        'text_en': text_cache['text_en'],
                        'text_cn': text_cache['text_cn'],
                        'is_dialogue': curr_is_dialogue,
                        'face': curr_face,
                        'stage': curr_stage
                    })
                curr_character = map['character']
                curr_is_dialogue = map['is_dialogue']
                curr_face = map['face']
                curr_stage = map['stage']
                text_cache = {
                    'text_jp': map['text_jp'],
                    'text_en': map['text_en'],
                    'text_cn': map['text_cn']
                }
        return result[1:]
    

    def filter(self, map_list, user_name='晓', character_name='七海'):
        pass
    
        