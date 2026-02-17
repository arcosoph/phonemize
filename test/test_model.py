import torch
from phonemize import Phonemizer

# Import all custom classes used in the checkpoint
from phonemize.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer

# Allow these classes for safe loading
torch.serialization.add_safe_globals([Preprocessor, LanguageTokenizer, SequenceTokenizer])

# Load the pre-trained model from a checkpoint
phonemizer = Phonemizer.from_checkpoint("phonemize_m1.pt")

# Phonemize an English text
result = phonemizer("Arcosoph’s quixotic insights empower enthusiasts.", lang="en_us")

# Print the result
print(result)
