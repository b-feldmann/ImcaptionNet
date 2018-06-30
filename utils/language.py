def predictions_to_captions(predictions, vocab):
  captions = []
  for i in range(predictions.shape[0]):  # for all images
    caption = []
    for j in range(predictions.shape[1]):  # for all elements in sequence
      word = vocab.get(predictions[i, j])
      if word:
        caption.append(word)
        if word == '<eos>':
          break
      else:
        caption.append('<unk>')
    captions.append(caption)

  return captions
