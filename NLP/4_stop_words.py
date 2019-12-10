# Implementing lemmatization
import spacy
from spacy.lang.en import English

nlp = English()

lem = nlp("run runs running runner")


# finding lemma for each word
for word in lem:
    print(word.text,word.lemma_)

