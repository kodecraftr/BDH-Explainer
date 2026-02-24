import os
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
# We will save the output files in a folder named 'tinystories_bin'
output_dir = 'tinystories_bin'
os.makedirs(output_dir, exist_ok=True)

# 1. Load the dataset (This will download it to your machine's cache first)
print("Downloading/Loading TinyStories...")
# We use the full dataset now since your local internet is likely more stable
dataset = load_dataset("roneneldan/TinyStories", split="train")

# 2. Setup Tokenizer (GPT-2 style, used by BDH/NanoGPT)
enc = tiktoken.get_encoding("gpt2")
EOS_TOKEN = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

def process(example):
    ids = enc.encode(example['text'])
    ids.append(EOS_TOKEN)
    return {'ids': ids, 'len': len(ids)}

# 3. Tokenize a subset (or all of it)
# Let's do 100,000 examples to be safe and quick. 
# Remove .select(...) if you want the FULL dataset (takes longer).
print("Tokenizing data...")
tokenized = dataset.select(range(100000)).map(
    process,
    remove_columns=['text'],
    desc="Processing texts",
    num_proc=4 # Uses 4 CPU cores to speed it up
)

# 4. Split and Save to Binary Files
split_idx = int(len(tokenized) * 0.9)
splits = {
    'train': tokenized.select(range(split_idx)),
    'val': tokenized.select(range(split_idx, len(tokenized)))
}

for split, dset in splits.items():
    filename = os.path.join(output_dir, f'{split}.bin')
    arr_len = np.sum(dset['len'])
    print(f"Writing {filename}...")
    
    # Create binary file
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    
    idx = 0
    for example in tqdm(dset):
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()

print(f"\nSUCCESS! Files are in the '{output_dir}' folder.")
print("Upload 'train.bin' and 'val.bin' to your Google Drive.")