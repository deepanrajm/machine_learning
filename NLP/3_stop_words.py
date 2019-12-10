#Stop words
#importing stop words from English language.
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

spacy_stopwords = STOP_WORDS

#Printing the total number of stop words:
print('Number of stop words: %d' % len(spacy_stopwords))

#Printing first ten stop words:
print('First ten stop words: %s' % list(spacy_stopwords)[:10])

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()


text = """When learning Artificial Intelligence, you shouldn't get discouraged!
Challenges and setbacks aren't failures, they're just part of the journey. You've got this!"""

#Implementation of stop words:
filtered_sent=[]

#  "nlp" Object is used to create documents with linguistic annotations.
doc = nlp(text)

# filtering stop words
for word in doc:
    if word.is_stop==False:
        filtered_sent.append(word)
print("Filtered Sentence:",filtered_sent)