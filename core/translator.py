from googletrans import Translator

class TextHelper:
    def __init__(self, config):
        self.translator = Translator()

    def translate(self, text, src_lang, dst_lang):
        print(text, ' ', src_lang, ' ', dst_lang)
        output = self.translator.translate(text, src=src_lang, dest=dst_lang)
        return output.text

# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

'''
import time
import requests
import random
import json
from hashlib import md5

class TextHelper:
    def __init__(self,config):
        
        # Set your own appid/appkey.
        self.appid = config["TS_APPID"]
        self.appkey = config['TS_APPKEY']
        self.human_trans = config["HUMAN_TRANS"]
    
    def lang_code_map(self,lang_code):
        lang_dict = {
            'en': "en",
            'es': "spa", 
            'fr': "fra", 
            'de': "de", 
            'it': "it",
            'pt': "pt", 
            'pl': "pl", 
            'tr': "tr", 
            'ru': "ru", 
            'nl': "nl", 
            'cs': "cs", 
            'ar': "ara", 
            'zh-cn': "zh", 
            'ja': "jp",
            'hu': "hu",
            'ko': "kor"
        }
        return lang_dict[lang_code]
    def translate(self, text, src_lang, dst_lang):
        # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
        src_lang = self.lang_code_map(src_lang)
        dst_lang = self.lang_code_map(dst_lang)
        
        print(text, ' ', src_lang, ' ', dst_lang)
        endpoint = 'http://api.fanyi.baidu.com'
        path = '/api/trans/vip/translate'
        url = endpoint + path

        salt = random.randint(32768, 65536)
        sign = self.make_md5(self.appid + text + str(salt) + self.appkey)

        # Build request
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': self.appid, 'q': text, 'from': src_lang, 'to': dst_lang, 'salt': salt, 'sign': sign}

        # Send request
        try:
            r = requests.post(url, params=payload, headers=headers)
            result = r.json()
            dst_text = result['trans_result'][0]['dst']
            print(json.dumps(result, indent=4, ensure_ascii=False))
        except Exception as e:
            dst_text = input("error, manual input of translation result:\n")
            return dst_text.strip()

        # Show response
        
        # print('translate result:{}'.format(result['trans_result'][0]['dst']))
        
        if self.human_trans == 1:
            print("human trans mode, you can edit it in config.json HUMAN_TRANS as 0 to disable")
            is_change = input("is tranlator good?input 1 is good else 0:\n")
            if int(is_change) == 0:
                dst_text = input("translate manual:\n")
                dst_text = dst_text.strip()
        else:
            print("You can edit it in config.json HUMAN_TRANS as 1 to enable human trans mode")
            
        print("final translate result:{}".format(dst_text))
        return dst_text
            

    # Generate salt and sign
    def make_md5(self,s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()



'''