# Import Library
from spacy.lang.xx import MultiLanguage
nlp = MultiLanguage()

from keras.preprocessing.sequence import pad_sequences

## Convert text to vector
def convert_to_vector(text, word2Idx, char2Idx, maxwordlength):
    """
        Args:
            text (str): Input sentence
        Return:
            A list consist of.
            [0]: A list of Token's index in the sentence
            [1]: A list of Casing condition of token in the sentence
            [2]: A list of list of token's Characters pattern
    """
    word_indices = []
    char_indices = []
    doc = nlp(text)
    for token in doc:
        i = token.text
        try:
            word_indices.append(word2Idx[i])
        except KeyError:
            word_indices.append(word2Idx["UNK"])
        
        tok = []
        for j in i:
            try:
                tok.append(char2Idx[j])
            except KeyError:
                tok.append(char2Idx["UNK"])
        char_indices.append(pad_sequences([tok], maxlen=maxwordlength)[0])
        
    return [word_indices, char_indices]

