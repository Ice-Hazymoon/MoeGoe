""" from https://github.com/keithito/tacotron """
from text import cleaners
import re


def text_to_sequence(text, symbols, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  _symbol_to_id = {s: i for i, s in enumerate(symbols)}

  sequence = []

  clean_text = _clean_text(text, cleaner_names)
  # clean_text = re.sub(r"\[[^\]]*\]", "", clean_text) # remove [xxx] in text
  # clean_text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", clean_text) # remove @xxx, non-ascii, url, rt, http
  print(clean_text)
  for symbol in clean_text.split(" "):
    if symbol in _symbol_to_id:
      sequence.append(_symbol_to_id[symbol])
    else:
      for s in symbol:
        if s in _symbol_to_id:
          sequence.append(_symbol_to_id[s])
    sequence.append(_symbol_to_id[" "])
  if sequence[-1] == _symbol_to_id[" "]:
    sequence = sequence[:-1]
  return sequence


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
