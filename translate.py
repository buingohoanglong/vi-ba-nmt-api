from transformer_based.translate import translate as old_translate
from transformerbertpgn.translate import translate as new_translate

TRANSFORMER = 'transformer'
BERT_FUSED = 'bert-fused'
TBMP = 'tbmp'

def translate(text, selected_model):
    ba = ''
    if selected_model == TRANSFORMER:
        ba = old_translate(text)
    else:
        ba = new_translate(text, selected_model)
    return ba
