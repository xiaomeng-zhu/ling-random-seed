import os, logging, argparse, math, pickle, gc, random
import torch
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import datasets
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm
import numpy as np

# Use default GPT2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(random_seed, epoch):
    model = GPT2LMHeadModel.from_pretrained(f"model_{random_seed}_epoch{epoch}")
    model.to(DEVICE)
    model.eval()
    logging.info(f"Loading model from model_{random_seed}_epoch{epoch}")
    return model

def gpt2_collate_fn(batch):
    """Pads batch to the maximum length in the batch dynamically."""
    input_ids = []
    attention_masks = []
    for item in batch:
        input_ids.append(item["input_ids"])
        attention_masks.append(item["attention_mask"])

    # Find max length in this batch
    max_length = max(len(ids) for ids in input_ids)
    
    # Pad sequences in the batch to max_length
    input_ids_padded = torch.stack([torch.cat([ids, torch.full((max_length - len(ids),), tokenizer.eos_token_id)]) for ids in input_ids])
    attention_mask_padded = torch.stack([torch.cat([ams, torch.full((max_length - len(ams),), 0)]) for ams in attention_masks])

    # print(f"Input_ids {input_ids_padded[0]}; attention_masks {attention_mask_padded[0]}")

    return {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded}

def test(model):
    """Run testing on a separate test file."""
    data_files = {
            "train": [
                "childes_train.txt", "bnc_spoken_train.txt", "gutenberg_train.txt",
                "open_subtitles_train.txt", "simple_wiki_train.txt", "switchboard_train.txt"
            ],
            "dev": [
                "childes_dev.txt", "bnc_spoken_dev.txt", "gutenberg_dev.txt",
                "open_subtitles_dev.txt", "simple_wiki_dev.txt", "switchboard_dev.txt"
            ],
            "test": [
                "childes_test.txt", "bnc_spoken_test.txt", "gutenberg_test.txt",
                "open_subtitles_test.txt", "simple_wiki_test.txt", "switchboard_test.txt"
            ]
        }
    
    dataset = load_dataset("xmzhu/babylm-strict-small", data_files = data_files)
    test_data = dataset["test"].map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=gpt2_collate_fn)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            # Forward pass
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
            logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
            # Shift labels to align with prediction
            shift_logits = logits[:, :-1, :].contiguous() # shape: (batch_size, seq_length-1, vocab_size) # from 0th to last-1
            shift_labels = inputs[:, 1:].contiguous() # shape (batchs_size, seq_length-1) # from 1st to last

            # flatten the tensors
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            

            test_loss += loss.item()
            gc.collect()
            torch.cuda.empty_cache()

    avg_test_loss = test_loss / len(test_loader)
    
    logging.info(f"Average test loss: {avg_test_loss:.4f}")

    return avg_test_loss

if __name__ == "__main__":
    seeds = [5345, 7445, 1732, 8720, 4522, 7454, 577, 7429, 5578, 440, 2751, 5731, 5272, 5653, 4000, 4557, 583, 6290, 7051, 4895]
    for s in seeds:
        model = load_model(s, 0)
        avg_test_loss = test(model)
        logging.info(f"Average test loss for random seed {s}: {avg_test_loss:.4f}")