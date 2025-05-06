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
from lr_scheduler import *

# ALL_VERBS_DICT = get_all_verbs()

parser = argparse.ArgumentParser()

# Tokenizer Parameters
parser.add_argument("--tokenizer_dir", type=str)

# Training Parameters
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--train_batch_size", type=int, default=256)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--eval_steps", help="Number of training steps to go between evaluations", type=int, default=1000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--dropout", help="Dropout", type=float, default=0.1)
parser.add_argument("--learning_rate", help="Learning rate", type=float, default=0.0001)
parser.add_argument("--n_epochs", help="Number of training epochs", type=int, default=5)
parser.add_argument("--lr_scheduler_type", help="Learning rate scheduler type (cosine or linear)", type=str, default="linear")
parser.add_argument("--warmup_steps", help="Number of steps that are warmup", type=int, default=0)
parser.add_argument("--logging_steps", help="Number of steps to log training progress", type=int, default=100)

# Logging Parameters
parser.add_argument("--output_dir", help="Directory to save model weights in", type=str)
parser.add_argument("--log_dir", help="Directory to save logs in", type=str)


args = parser.parse_args()

# Set up logging
log_file_name = f"{args.model_name}-{args.random_seed}"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(),logging.FileHandler("logs/" + log_file_name + ".log")])
logging = logging.getLogger(__name__)
logging.info(args)

current_seed = args.random_seed
random.seed(current_seed)
np.random.seed(current_seed)
torch.manual_seed(current_seed)
logging.info(f"Random seed set to {current_seed}")
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        logging.info(f"Cuda random seed set to {current_seed}")
        torch.cuda.manual_seed(current_seed)


# Use default GPT2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

class BabyLMDataset:
    def __init__(self, tokenizer, block_size):
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
        
        dataset = load_dataset("xmzhu/baby-lm", data_files = data_files)

        self.tokenizer = tokenizer
        
        # We use batched matching to speed up processing
        train_ds = dataset["train"].map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        dev_ds = dataset["dev"].map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
        dev_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        test_ds = dataset["test"].map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
        test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        self.train = train_ds
        self.dev = dev_ds
        self.test = test_ds

    def __len__(self):
        return len(self.train) + len(self.dev) + len(self.test)

    def __getitem__(self, split, i):
        if split == "train":
            return self.train[i]
        elif split == "dev":
            return self.dev[i]
        elif split == "test":
            return self.test[i]
        else:
            print("Error: split is not defined")

logging.info("Loading BabyLM Dataset...")
dataset = BabyLMDataset(tokenizer = tokenizer,
                         block_size=128)
logging.info("Finished loading BabyLM")


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




class ModelTrainer:
    def __init__(self, tokenizer, model_name, data, n_epochs, train_batch_size, eval_batch_size,
                 warmup_steps, weight_decay, eval_steps, save_steps, logging_steps, 
                 learning_rate, lr_scheduler_type,
                 output_dir):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.n_epochs = n_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type

        # Load Data
        self.train_data = data.train
        self.dev_data = data.dev
        self.test_data = data.test

        
        # Define model architecture
        self.config = AutoConfig.from_pretrained(model_name,
                                            vocab_size=len(self.tokenizer),
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id
                                            )
        self.model = GPT2LMHeadModel(self.config)
        self.collate_fn = gpt2_collate_fn
        logging.info(f"Using gpt2_collate_fn with bos_token_id {tokenizer.bos_token_id} and eos_token_id {tokenizer.eos_token_id}")

        self.model.to(self.device)
        self.model_size = sum(t.numel() for t in self.model.parameters())
        logging.info(f"Model size: {self.model_size/1000**2:.1f}M parameters")

        self.output_dir = output_dir
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.scheduler = get_scheduler(
                self.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.n_epochs)

    
    def train(self):
        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            evaluation_strategy="steps",
            eval_steps=self.eval_steps,
            logging_steps=self.logging_steps,
            num_train_epochs=self.n_epochs,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            learning_rate=self.learning_rate,
            save_steps=self.save_steps,
            #use_mps_device=True, # enable when training on Mac with Apple Silicon
            )
        
        self.model.train() # put model in training mode
        train_loader = DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=False, collate_fn=self.collate_fn)

        num_step = 0

        for epoch in range(self.n_epochs):
            logging.info(f"Epoch {epoch + 1}/{self.n_epochs}")
            epoch_loss = 0
            for batch in tqdm(train_loader, desc="Training"):
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)  
                
                # Shift labels to align with prediction
                outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=inputs) # logits is of size batch_size, seq_length, vocab_size         
                logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
                shift_logits = logits[:, :-1, :].contiguous() # shape: (batch_size, seq_length-1, vocab_size) # from 0th to last-1
                shift_labels = inputs[:, 1:].contiguous() # shape (batchs_size, seq_length-1) # from 1st to last

                # define loss function
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # print(f"loss is {loss.item()}")
               
                epoch_loss += loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                gc.collect()

                # record number of steps
                num_step += 1
                if num_step % self.logging_steps == 0:
                    logging.info(f"Step {num_step}: Loss: {loss.item():.4f}, Learning rate: {self.scheduler.get_last_lr()[0]:.4f}")

            avg_epoch_loss = epoch_loss / len(train_loader)
            perplexity = math.exp(avg_epoch_loss) if avg_epoch_loss < 20 else float('inf')  # Avoid overflow for large losses
            logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}, Perplexity: {perplexity:.4f}, Learning rate: {self.scheduler.get_last_lr()[0]:.4f}")

            self.scheduler.step()

            # Optional: Evaluate after each epoch
            if self.test_data:
                self.evaluate(epoch)
                self.save_model(epoch)
            

    def evaluate(self):
        """Run testing on a separate test file."""
        val_loader = DataLoader(self.dev_data, batch_size=self.eval_batch_size, shuffle=False, collate_fn=self.collate_fn)
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
                logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
                # Shift labels to align with prediction
                shift_logits = logits[:, :-1, :].contiguous() # shape: (batch_size, seq_length-1, vocab_size) # from 0th to last-1
                shift_labels = inputs[:, 1:].contiguous() # shape (batchs_size, seq_length-1) # from 1st to last

                # flatten the tensors
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                

                val_loss += loss.item()
                gc.collect()

        avg_val_loss = val_loss / len(val_loader)
        
        logging.info(f"Average val loss: {avg_val_loss:.4f}")

        return avg_val_loss

    def test(self):
        """Run testing on a separate test file."""
        test_loader = DataLoader(self.test_data, batch_size=self.eval_batch_size, shuffle=False, collate_fn=self.collate_fn)
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
                logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
                # Shift labels to align with prediction
                shift_logits = logits[:, :-1, :].contiguous() # shape: (batch_size, seq_length-1, vocab_size) # from 0th to last-1
                shift_labels = inputs[:, 1:].contiguous() # shape (batchs_size, seq_length-1) # from 1st to last

                # flatten the tensors
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                

                test_loss += loss.item()
                gc.collect()

        avg_test_loss = test_loss / len(test_loader)
        
        logging.info(f"Average test loss: {avg_test_loss:.4f}")

        return avg_test_loss

    
    
    def save_model(self, epoch):
        """Save the model and tokenizer."""
        logging.info(f"Saving model and tokenizer to {self.output_dir}")
        self.model.save_pretrained(f"{self.output_dir}_epoch{epoch}")
    
model_trainer = ModelTrainer(tokenizer=tokenizer,
                             model_name=args.model_name,
                             data=dataset,
                             n_epochs=args.n_epochs,
                             train_batch_size=args.train_batch_size,
                             eval_batch_size=args.eval_batch_size,
                             warmup_steps=args.warmup_steps,
                             weight_decay=args.weight_decay,
                             eval_steps=args.eval_steps,
                             save_steps=args.save_steps,
                             logging_steps=args.logging_steps,
                             learning_rate=args.learning_rate,
                             lr_scheduler_type=args.lr_scheduler_type,
                             output_dir=args.output_dir
                            )
model_trainer.train()
model_trainer.test()

