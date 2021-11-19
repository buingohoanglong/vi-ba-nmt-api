import warnings
from collections import Counter

from nltk import ngrams
warnings.filterwarnings('ignore')
import nltk.translate.bleu_score as bleu


def calculateBleuScore(input_dir="data", output_dir="bleu_calculation", ref_corpus="", candidate_corpus=""):
    txtRefFile = open(input_dir + "/" + ref_corpus, encoding='utf-8', errors='ignore').read()
    txtCandidateFile = open(input_dir + '/' + candidate_corpus, encoding='utf-8', errors='ignore').read()
    txtRefFile = txtRefFile.replace("!\n", "\n").replace("?\n", "\n").replace(".\n", "\n").replace(":\n", "\n")
    txtRefFile = txtRefFile.replace("(", "").replace(")", "")

    txtCandidateFile = txtCandidateFile.replace("!\n", "\n").replace("?\n", "\n").replace(".\n", "\n").replace(":\n",                                                                                                           "\n")
    txtCandidateFile = txtCandidateFile.replace(" đ c ", "đ/c")
    splitsRef = txtRefFile.split('\n')
    splitsCandidate = txtCandidateFile.split('\n')
    print(splitsRef[1].split())
    print(splitsCandidate[1].split())
    # score = bleu_score(splitsCandidate[0].split(), splitsRef[0].split())
    # score = bleu.sentence_bleu([splitsRef[2].split()], splitsCandidate[2].split())
    # hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which']
    # reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that']
    # reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which']
    # score = bleu.sentence_bleu([reference1, reference2], hypothesis1)
    # print('calculateBleuScore: ', score)
    # temp = list(map(lambda ref: [ref.split()], splitsRef))
    # print(temp)
    splC = list(map(lambda candidate: candidate.split(), splitsCandidate))
    score = bleu.corpus_bleu(list(map(lambda ref: [ref.split()], splitsRef)), splC)
    print('Bleu score with 200 samples is ', score)

    with open(input_dir + '/' + ref_corpus + '.t', 'w', encoding='utf-8') as f:
        f.write(txtRefFile)


# ttt = postProcessing(src_sentence="Bố đang đan bảy cái nữa.", translate_sentence="bă atung wă tanh tơpơh tŏ piêu. ....")
# print('post processing: ', ttt)
calculateBleuScore(input_dir='new_output_1', ref_corpus="truth-sentences-bana-0507.txt",
                   candidate_corpus="test-sentences-bana-0507.txt")