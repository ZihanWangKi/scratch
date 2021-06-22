import collections
import os
import unicodedata
from transformers.tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


import random
import numpy as np
import torch
from tqdm import tqdm


class CharBertTokenizer:
    def __init__(self, max_char_position_embeddings=20, max_word_position_embeddings=128):
        # <pad> <bos> <eos> <bow> <eow> <unk> <mask>
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.bow_token_id = 3
        self.eow_token_id = 4
        self.unk_token_id = 5
        self.mask_token_id = 6

        self.id_to_char = {0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<bow>', 4: '<eow>', 5: '<unk>', 6: '<mask>'}
        for ascii_i in range(33, 127):
            ascii_c = chr(ascii_i)
            self.id_to_char[ascii_i] = ascii_c
        self.char_to_id = {
            v: k for k, v in self.id_to_char.items()
        }
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.max_chars_in_word = max_char_position_embeddings
        self.max_words_in_sentence = max_word_position_embeddings

        self.bos_word = [self.bow_token_id, self.bos_token_id, self.eow_token_id]
        self.eos_word = [self.bow_token_id, self.eos_token_id, self.eow_token_id]
        self.mask_word = [self.bow_token_id, self.mask_token_id, self.eow_token_id]
        self.pad_word = [self.pad_token_id] * self.max_chars_in_word

    def decode_id(self, id):
        assert isinstance(id, int)
        return self.id_to_char.get(id, '¿')

    def decode_word_ids(self, word_ids, char_att_mask=None):
        n_chars = len(word_ids)
        if char_att_mask is None:
            char_att_mask = [1 for _ in range(n_chars)]
        return "".join([self.decode_id(id) for id, char_mask in zip(word_ids, char_att_mask) if char_mask == 1])

    def decode_sentence_ids(self, sentence_ids, char_attention_mask=None, word_attention_mask=None):
        n_words = len(sentence_ids)
        n_chars = len(sentence_ids[0])
        if char_attention_mask is None:
            char_attention_mask = [[1 for _ in range(n_chars)] for _ in range(n_words)]
        if word_attention_mask is None:
            word_attention_mask = [1 for _ in range(n_words)]
        return " ".join([self.decode_word_ids(word_ids, char_att_mask) for
                         word_ids, word_att_mask, char_att_mask in
                         zip(sentence_ids, word_attention_mask, char_attention_mask) if word_att_mask == 1])

    def tokenize_word(self, word):
        if len(word) > self.max_chars_in_word - 2:
            word = word[: self.max_chars_in_word - 2]
        return [self.bow_token_id] + [self.char_to_id.get(c, self.unk_token_id) for c in word] + [self.eow_token_id]

    def tokenize_sentence(self, sentence):
        words = self.basic_tokenizer.tokenize(sentence)
        if len(words) > self.max_words_in_sentence - 2:
            words = words[: self.max_words_in_sentence - 2]
        return [self.bos_word] + [self.tokenize_word(word) for word in words] + [self.eos_word]

    def tokenize(self, sentences):
        input_ids = []
        for sentence in sentences:
            input_ids.append(self.tokenize_sentence(sentence))
        return {
            "input_ids": input_ids,
        }

    def tokenize_sentence_pair(self, sentence_1, sentence_2=None):
        assert isinstance(sentence_1, str), sentence_1
        assert sentence_2 is None or isinstance(sentence_1, str), sentence_2
        if sentence_2 is None:
            return self.tokenize_sentence(sentence_1)
        words_1 = self.basic_tokenizer.tokenize(sentence_1)
        words_2 = self.basic_tokenizer.tokenize(sentence_2)
        if len(words_1) > self.max_words_in_sentence // 2 - 2:
            words_1 = words_1[: self.max_words_in_sentence // 2 - 2]
        if len(words_2) > self.max_words_in_sentence // 2 - 2:
            words_2 = words_2[: self.max_words_in_sentence // 2 - 2]
        return [self.bos_word] + [self.tokenize_word(word) for word in words_1] + [self.eos_word] + \
               [self.bos_word] + [self.tokenize_word(word) for word in words_2] + [self.eos_word]

    def tokenize_for_classification(self, sentences_1, sentences_2=None):
        input_ids = []
        for i in range(len(sentences_1)):
            input_ids.append(self.tokenize_sentence_pair(sentences_1[i], sentences_2[i] if sentences_2 is not None else None))
        return {
            "input_ids": input_ids,
        }

    def padding(self, input_ids):
        padded_input_ids = []
        char_attention_mask = []
        word_attention_mask = []
        for word_ids in input_ids:
            length = len(word_ids)
            char_attention_mask.append([1] * length + [0] * (self.max_chars_in_word - length))
            padded_input_ids.append(word_ids + [self.pad_token_id] * (self.max_chars_in_word - length))
            word_attention_mask.append(1)

        for _ in range(self.max_words_in_sentence - len(input_ids)):
            padded_input_ids.append(self.pad_word)
            char_attention_mask.append([0] * self.max_chars_in_word)
            word_attention_mask.append(0)

        return {
            "padded_input_ids": padded_input_ids,
            "char_attention_mask": char_attention_mask,
            "word_attention_mask": word_attention_mask
        }

    def perform_padding_for_classification(self, batched_input):
        input_ids = []
        char_attention_mask = []
        word_attention_mask = []
        decoder_input_ids = []
        decoder_attention_mask = []
        labels = []
        n_batches = len(batched_input)
        for batch_no in range(n_batches):
            batch_input_ids = batched_input[batch_no]["input_ids"]
            shifted_batch_input_ids = [
                [word_ids[-1]] + word_ids[: -1] for word_ids in batch_input_ids
            ]
            padded = self.padding(shifted_batch_input_ids)
            decoder_input_ids.append(padded["padded_input_ids"])
            decoder_attention_mask.append(padded["char_attention_mask"])

            padded = self.padding(batch_input_ids)
            input_ids.append(padded["padded_input_ids"])
            char_attention_mask.append(padded["char_attention_mask"])
            word_attention_mask.append(padded["word_attention_mask"])
            labels.append(batched_input[batch_no]["label"])

        input_ids = torch.tensor(input_ids)
        char_attention_mask = torch.tensor(char_attention_mask)
        word_attention_mask = torch.tensor(word_attention_mask)
        decoder_input_ids = torch.tensor(decoder_input_ids)
        decoder_attention_mask = torch.tensor(decoder_attention_mask)
        labels = torch.tensor(labels)

        return {
            "input_ids": input_ids,
            "char_attention_mask": char_attention_mask,
            "word_attention_mask": word_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }

    def perform_masking(self, batched_input, apply_noise_prob=0.2, mask_word_cutoff=0.3, shuffle_cutoff=0.6, swap_cutoff=0.9):
        input_ids = []
        char_attention_mask = []
        word_attention_mask = []
        decoder_input_ids = []
        decoder_attention_mask = []
        labels = []
        n_batches = len(batched_input)
        for batch_no in range(n_batches):
            batch_input_ids = batched_input[batch_no]["input_ids"]
            shifted_batch_input_ids = [
                [word_ids[-1]] + word_ids[: -1] for word_ids in batch_input_ids
            ]
            padded = self.padding(shifted_batch_input_ids)
            decoder_input_ids.append(padded["padded_input_ids"])
            decoder_attention_mask.append(padded["char_attention_mask"])

            batch_labels = [[-100 for _ in range(self.max_chars_in_word)] for _ in range(self.max_words_in_sentence)]

            noised_input_ids = []
            n_words = len(batch_input_ids)
            for word_no in range(n_words):
                word_ids = batch_input_ids[word_no][:]  # copy!
                if word_ids[1] not in [self.bos_token_id, self.eos_token_id] and random.uniform(0,
                                                                                                1) <= apply_noise_prob:
                    n_chars = len(word_ids)
                    for char_no in range(n_chars):
                        batch_labels[word_no][char_no] = word_ids[char_no]
                    indicator = random.uniform(0, 1)
                    if indicator <= mask_word_cutoff:
                        word_ids = self.mask_word
                    elif indicator <= shuffle_cutoff:
                        chars = word_ids[1: -1]
                        random.shuffle(chars)
                        while len(chars) > 0:
                            if random.uniform(0, 1) <= 0.5:
                                chars = chars[: -1]
                            else:
                                break
                        while len(chars) < self.max_chars_in_word - 2:
                            if random.uniform(0, 1) <= 0.5:
                                chars.append(random.randint(33, 126))
                            else:
                                break
                        word_ids = [word_ids[0]] + chars + [word_ids[-1]]
                    elif indicator <= swap_cutoff:
                        prob = 1.0
                        while True:
                            if random.uniform(0, 1) <= prob:
                                index_i = random.randint(1, n_chars - 2)
                                index_j = random.randint(1, n_chars - 2)
                                word_ids[index_i], word_ids[index_j] = word_ids[index_j], word_ids[index_i]
                                prob = 0.5
                            else:
                                break
                noised_input_ids.append(word_ids)
            padded = self.padding(noised_input_ids)
            input_ids.append(padded["padded_input_ids"])
            char_attention_mask.append(padded["char_attention_mask"])
            word_attention_mask.append(padded["word_attention_mask"])
            labels.append(batch_labels)

        input_ids = torch.tensor(input_ids)
        char_attention_mask = torch.tensor(char_attention_mask)
        word_attention_mask = torch.tensor(word_attention_mask)
        decoder_input_ids = torch.tensor(decoder_input_ids)
        decoder_attention_mask = torch.tensor(decoder_attention_mask)
        labels = torch.tensor(labels)

        return {
            "input_ids": input_ids,
            "char_attention_mask": char_attention_mask,
            "word_attention_mask": word_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }


if __name__ == '__main__':
    tokenizer = CharBertTokenizer()
    print(tokenizer.tokenize_sentence("I went for lunch"))
    pass
