# Word tokenization
from spacy.lang.en import English

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

text = """When learning Artificial Intelligence, you shouldn't get discouraged!
Challenges and setbacks aren't failures, they're just part of the journey. You've got this!"""

#  "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text)
print ("Document format ----> ", my_doc)
# Create list of word tokens
token_list = []
for token in my_doc:
	token_list.append(token.text)
print("List Format ----> ",token_list)