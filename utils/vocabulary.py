def captions_to_tokens(anns_list, tokenizer):
  """
    Args:
      anns_list: A list of COCO object anns dictionaries.
  """
  print('Tokenizing & Counting words in captions & adding <start>, <eos> tags')
  if tokenizer == 'neuraltalk':
    import string
    translator = str.maketrans('', '', string.punctuation)
  counts = {}
  for anns in anns_list:
    for ann_id in list(anns.keys()):
      caption = anns[ann_id]['caption']
      if tokenizer == 'nltk':
        import nltk
        nltk.download('punkt', quiet=True)
        tokens = nltk.tokenize.word_tokenize(str(caption))
      elif tokenizer == 'neuraltalk':
        tokens = str(caption).lower().translate(translator).strip().split()
      tokens = [t.lower() for t in tokens]
      tokens = ['<start>'] + tokens + ['<eos>']
      for token in tokens:
        if token not in counts:
          counts[token] = 0
        counts[token] += 1
      anns[ann_id]['caption'] = tokens
  return counts


def caption_tokens_to_id(anns_list, vocab, counts, word_count_threshold):
  """Transform tokenized captions to ids.
    NOTE: Uses 1 index vocabulary mapping. Therefore actual vocabulary size is
    + 1!
    Args:
      anns_list: A list of COCO object anns dictionaries.
      vocab: A list of words acting as the vocabulary.
      counts: The output of captions_to_tokens().
      word_count_threshold: Words with count lower than this will be replaced
        with a special UNK symbol.
  """
  print('Transforming tokenized captions to indices')
  # Transform captions to indices
  wtoi = {w: i + 1 for i, w in enumerate(vocab)}
  for anns in anns_list:
    for ann_id in list(anns.keys()):
      caption = anns[ann_id]['caption']
      for i, token in enumerate(caption):
        if counts[token] <= word_count_threshold:
          caption[i] = wtoi['UNK']
        else:
          caption[i] = wtoi[token]


def preprocess_captions(anns_list, word_count_threshold, tokenizer):
  print('Processing captions for {} COCO annotation objects'.format(
      len(anns_list)))
  counts = captions_to_tokens(anns_list, tokenizer)
  # Create vocabulary
  vocab = [w for w, n in counts.items() if n > word_count_threshold]
  # Determine UNK words
  bad_words = [w for w, n in counts.items() if n <= word_count_threshold]
  bad_count = sum(counts[w] for w in bad_words)
  if bad_count > 0:
    print('Adding UNK token to vocabulary for infrequent words')
    vocab.append('UNK')

  # Statistics
  print('Statistics for {} COCO annotation objects'.format(len(anns_list)))
  cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
  print('Top words and their counts:')
  print('\n'.join(map(str, cw[:20])))
  total_words = sum(counts.values())
  print('Total words:', total_words)
  print('Number of words < threshold: {}/{} = {:.2f}'.format(
      len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
  print('Number of UNKs: {}/{} = {:.2f}'.format(
      bad_count, total_words, bad_count * 100.0 / total_words))
  print('Number of words in vocabulary: {}'.format(len(vocab)))

  caption_tokens_to_id(anns_list, vocab, counts, word_count_threshold)

  return vocab, len(vocab) + 1, counts
