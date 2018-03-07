#
# Generate rubric vector
#
#
import numpy as np
import fastText
from gensim.models.wrappers.fasttext import FastText
from gensim.models import Word2Vec
from scipy import spatial

def cossim_np(a, b):
    return 1 - spatial.distance.cosine(a, b)


# can support OOV word output.
def get_sent_vec(sent):
    return model.get_sentence_vector(sent)


# model[word] will raise error if word is not in vocab
# have to use words in the vocab
def get_sent_vec_w2v(sent):
    vecs = [model[word] for word in sent.split() if word in model.wv.vocab]
    return np.mean(vecs, axis=0)


def maxsim_on_a_note(note, utts):
    # vec_note = get_sent_vec(note)
    # return np.max([cossim_np(vec_note, get_sent_vec(utt)) for utt in utts])
    cossim = []
    for utt in utts:
        y = model.wmdistance(note.lower(), utt.lower())
        print('{}\t{}\t{}'.format(note, utt, y))
        cossim.append(y)
    print('\n')
    return np.min(cossim)


if __name__ == '__main__':
    sa = "In order to replicate this experiment, there is quite a few other pieces of info that they need. First, you need to know how vinegar you have to pour on each sample. Secondly you need to know how much of each sample you need. Finally, you need to know the mass of the cups so you don't mix up the sample weight. That's the information you need to replicate this expiriment."
    sa_3 = "The procedure is unclear. In order to effectively replicate this experiment. I would need to know what the four samples that are needed, were, i would also need to know how much vinegar to pour into each container. It would help to know how  large each container was as well.^p1.vinegar amount. 2. types of sampling. 3. containers size."

    sa_1 = "To replicate the experiment you would need to find out exactly what samples were tested in this case. The samples are marble, limestone, wood and plastic. You would need to find out when the experiment was being conducted and whose you're testing."
    notes = ['how much vinegar in each container',
             'what type of vinegar in each container',
             'what materials',
             'what size of materials',
             'what surface area of materials',
             'how long each sample was rinsed in distilled water',
             'what drying method',
             'what size or type of container']

    utts = sa_1.split(sep='.')[:-1] # drop the last ''
    # model = fastText.load_model('/Users/lchen/Documents/GitHub/call2018/data/fastText/m#
    model = FastText.load_fasttext_format('/Users/lchen/Documents/GitHub/call2018/data/fastText/model/wiki.simple')
    # model = Word2Vec.load('/Users/lchen/Documents/macbook/github/CNN-Sentence-Classifier/embd/glove.6B.300d.txt')
    rubvec = [maxsim_on_a_note(note, utts) for note in notes]
    print(rubvec)





