import spacy
import re
# from underthesea import word_tokenize
from pyvi import ViTokenizer, ViPosTagger

class tokenize(object):
    
    def __init__(self, lang):
        self._lang = lang
        if lang != 'vi':
            if lang == 'en':
                self.nlp = spacy.load("en_core_web_sm")
            else:
                self.nlp = spacy.load(lang)
            
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        # sentence = sentence.lower()
        # sentence je ne peux pas accepter ça.
        # tokens ['je', 'ne', 'peux', 'pas', 'accepter', 'ça', '.']
        # print("sentence", sentence)
        if self._lang != 'vi':
            tokens = [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
        else:
            tokens = ViTokenizer.tokenize(sentence).split(' ')
            # tokens = word_tokenize(sentence, format="text")
        # print("tokens", tokens)
        return tokens
