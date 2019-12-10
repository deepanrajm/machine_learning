
# Word vector representation

import en_core_web_sm
nlp = en_core_web_sm.load()

mango = nlp(u'mango')

print(mango.vector.shape)
print(mango.vector)