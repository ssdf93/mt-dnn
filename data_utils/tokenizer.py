from DeBERTa.deberta import GPT2Tokenizer

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class DeBERTaTokenizer():
    def __init__(self, **kwargs):
        self.tokenizer = GPT2Tokenizer()

    @staticmethod
    def from_pretrained(model=None, do_lower_case=None):
        return DeBERTaTokenizer()

    def encode_plus(self,
        text,
        text_pair = None,
        add_special_tokens = True,
        max_length = None,
    ):
        first_tokens = self.tokenizer.tokenize(text)
        second_tokens = self.tokenizer.tokenize(text_pair) if text_pair is not None else None
        if second_tokens:
            _truncate_seq_pair(first_tokens, second_tokens, max_length - 3)
            tokens = ['[CLS]'] + first_tokens + ['[SEP]'] + second_tokens + ['[SEP]']
            token_types = [0] * (len(first_tokens) + 2) + [1] * (len(second_tokens) + 1)
        else:
            tokens = ['[CLS]'] + first_tokens[:max_length - 2] + ['[SEP]']
            token_types = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return {"input_ids": input_ids, "token_type_ids": token_types}
