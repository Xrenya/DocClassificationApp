import re

import nltk
from natasha import (Doc, MorphVocab, NamesExtractor, NewsEmbedding,
                     NewsMorphTagger, NewsNERTagger, NewsSyntaxParser,
                     Segmenter)
from nltk.corpus import stopwords

nltk.download('stopwords')


class TextCleaner:

    def __init__(self, lemma: bool = True):
        self.lemma = lemma
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)
        syntax_parser = NewsSyntaxParser(emb)
        ner_tagger = NewsNERTagger(emb)
        names_extractor = NamesExtractor(self.morph_vocab)
        self.en_stops = stopwords.words('english')
        self.ru_stops = stopwords.words('russian')
        self.punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        self.words_pattern = '[а-я]+'

    def execute(self, text):
        text = self.text_preprocessing(text)
        if self.lemma:
            text = self.lemmatize(text)
        return text

    def text_preprocessing(self, data):
        data = " ".join(x.lower() for x in data.split())
        data = data.replace('[^\w\s]', '')
        data = " ".join(x for x in data.split()
                        if x not in self.ru_stops and x not in self.en_stops)
        for punc in self.punc:
            if punc in data:
                data = data.replace(punc, "")
        data = re.sub(' +', ' ', data)
        return " ".join(
            re.findall(self.words_pattern, data, flags=re.IGNORECASE))

    def lemmatize(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        tokens = []
        for token in doc.tokens:
            tokens.append(token.lemma)
        return " ".join(tokens)
