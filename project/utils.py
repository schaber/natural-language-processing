import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    # 'WORD_EMBEDDINGS': 'word_embeddings.tsv',
    'WORD_EMBEDDINGS': '../week3/modelSaveFile.tsv'
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    ########################
    embeddings = {}
    for line in open(embeddings_path):
        w, *vec = line.strip().split('\t')
        embeddings[w] = np.fromstring(','.join(vec), sep=',', dtype=np.float32)

    embeddings_dim = len(list(embeddings.values())[0])

    return embeddings, embeddings_dim


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function in the 3rd assignment.

    ########################
    #### YOUR CODE HERE ####
    ########################

    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    words = question.split(' ')
    # vectors = np.array([embeddings[w] for w in (words & set(embeddings.vocab))])
    vectors = []
    for w in words:
        try:
            vectors.append(embeddings[w])
        except KeyError:
            pass
    vectors = np.array(vectors)
    if not vectors.shape[0]:
        vectors = np.zeros(dim)
    if len(vectors.shape) > 1:
        ret_val = np.mean(vectors, axis=0)
    else:
        ret_val = vectors
    return ret_val


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
