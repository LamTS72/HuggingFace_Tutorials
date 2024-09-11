from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    pipeline,
)
import evaluate
import numpy as np
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class MaskedLM():
    def __init__(self):
        self.model = None
        self.data_collator = None
        self.raw_data = None
        self.model_checkpoint = None
        self.tokenized_dataset = None
        self.chunk_size = 128
        self.chunks_dataset = None
        self.split_dataset = None
        self.args = None

    def load_dataset(self, name="imdb"):
        self.raw_data = load_dataset(name)
        print("Name of dataset: ", name)
        print(self.raw_data)
    
    def load_support(self, mlm_probability=0.15, name="distilbert-base-uncased"):
        self.model_checkpoint = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=mlm_probability)
        print("Name of model checkpoint: " + self.model_checkpoint)
        print("Tokenizer Fast: ", self.tokenizer.is_fast)
        print("Symbol of masked word after tokenizer: ", self.tokenizer.mask_token)
        print("Model max length tokenizer: ", self.tokenizer.model_max_length)
        
    def explore_infoModel(self, k=5):
        model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint)
        model_parameters = model.num_parameters() / 1000000
        print(f">>> Number of parameters of {self.model_checkpoint}: {round(model_parameters)}M")

        example = "This is a great [MASK]."
        print("\n")
        print(">>> Example: ", example)
        inputs = self.tokenizer(example, return_tensors='pt')
        token_logits = model(**inputs).logits
        print(f"{'Number of tokens: ':<{30}}{ len(inputs.tokens())}")
        print(f"{'Tokens prepare for training: ':<{30}}{ inputs.tokens()}")
        print(f"{'IDs of tokens: ':<{30}}{ inputs.input_ids}")
        print(f"{'Maked IDs of Model: ':<{30}}{self.tokenizer.mask_token_id}")
        print(f"{'Logits of example: ':<{30}}{token_logits}")
        print(f"{'Shape of logits: ':<{30}}{token_logits.size()}")

        '''Find POSITION of [MASK] => EXTRACT LOGIT'''
        mask_token_index = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]
        print(f"{'Position of masked token index: ':<{30}}{mask_token_index}")

        '''Find LOGIT of token in VOCAB suitable for [MASK]'''
        mask_token_logits = token_logits[0, mask_token_index, :]
        print(f"{'Logit of tokens in Vocab for [MASK]: ':<{30}}{mask_token_logits}")

        '''Choose TOP CANDIDATES for [MASK] with highest logits => TOP LOGITS + POSITION of token suitable for [MASK] in VOCAB'''
        top_k_values = torch.topk(mask_token_logits, k, dim=1).values[0].tolist()
        print(f"{'Top value of suitable token in Vocab: ':<{30}}{top_k_values }")
        top_k_tokens = torch.topk(mask_token_logits, k, dim=1).indices[0].tolist()
        print(f"{'Position of suitable token in Vocab: ':<{30}}{top_k_tokens}")

        '''Show TOP CANDIDATES'''
        for token in top_k_tokens:
            print(">>> ", example.replace(self.tokenizer.mask_token, self.tokenizer.decode([token])))


    def get_feature_items(self, set="train", index=0, feature="text"):
        return  None if self.raw_data[set][index][feature] is None or self.raw_data[set][index][feature] == 0 else self.raw_data[set][index][feature]
    
    def get_pair_items(self, set="train", index=0, feature1="text", feature2="label"):
        feature1 = self.get_feature_items(set, index, feature1)
        feature2 = self.get_feature_items(set, index, feature2)
        if feature2 is not None:
            line1 = ""
            line2 = ""
            for word, label in zip(feature1, feature2):
                line1 += str(word)
                line2 += str(label)
            return line1, line2
        
        return feature1, feature1 
    
    def get_tokenizer(self, set="train", index=0, feature="text"):
        inputs = self.tokenizer(self.get_feature_items(set, index, feature))
        return inputs.tokens(), inputs.word_ids()
    
    def tokenizer_dataset(self, example):
        inputs = self.tokenizer(example["text"])
        inputs["word_ids"] = [inputs.word_ids(i) for i in range(len(inputs["input_ids"]))]
        return inputs
    
    def map_tokenize_dataset(self):
        print("Start of processing dataset")
        self.tokenized_dataset = self.raw_data.map(self.tokenizer_dataset, batched=True, remove_columns=["text","label"] )
        print("Done mapping")
        print("Tokenized dataset: ", self.tokenized_dataset)
    
    def group_text_chunk(self, example):
        '''Group all of text'''
        concatenate_example = {k : sum(example[k], []) for k in example.keys()}

        '''Compute the length of all'''
        total_length = len(concatenate_example["input_ids"])
        '''Final length for chunk size'''
        total_length = (total_length // self.chunk_size) *self.chunk_size
    
        '''Divide into chunks with chunk size'''
        chunks = {
            k: [t[i: i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
            for k, t in concatenate_example.items()
        }

        '''Create LABELS column from INPUT_IDS'''
        chunks["labels"] = chunks["input_ids"].copy()
        return chunks

    def map_chunk_dataset(self):
       print("Start of processing dataset")
       self.chunks_dataset = self.tokenized_dataset.map(self.group_text_chunk, batched=True)
       print("Done mapping")
       print("Chunked dataset: ", self.chunks_dataset)

    def dataset_split(self, test_size=0.2):
        self.split_dataset = self.chunks_dataset["train"].train_test_split(
            test_size=test_size, seed=42
        )
        print("Preparing dataset: ", self.split_dataset)


    def create_model(self):
        print("Start creating model")
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint)
        print(self.model)

    def create_argumentTrainer(self, output_dir="fine_tuned_", eval_strategy="epoch", logging_strategy="epoch",
                               learning_rate=2e-5, num_train_epochs=20, weight_decay=0.01, batch_size=64,
                               save_strategy="epoch", push_to_hub=False, hub_model_id="", fp16=True):
        logging_steps = len(self.split_dataset["train"]) // batch_size
        self.args= TrainingArguments(
            use_cpu=True,
            output_dir=f"{output_dir}{self.model_checkpoint}",
            overwrite_output_dir=True,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            fp16=fp16,
            logging_steps=logging_steps
        )
        print("Arguments ready for training")
        return self.args
    
    def call_train(self, model_path="pretrained_model_", set_train="train", set_val="test", push_to_hub=False, save_local=False):
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.split_dataset[set_train],
            eval_dataset=self.split_dataset[set_val],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        eval_result1 = trainer.evaluate()
        print("Perplexity before of training: ", math.exp(eval_result1['eval_loss']))

        print("Start training")
        trainer.train()
        print("Done training")

        eval_result2 = trainer.evaluate()
        print("Perplexity after of training: ", math.exp(eval_result2['eval_loss']))

        if save_local:
            trainer.save_model(model_path+self.model_checkpoint)
            print("Done saving to local")
  
        if push_to_hub:
            trainer.push_to_hub(commit_message="Training complete")
            print("Done pushing push to hub")

    def call_pipeline(self, local=False, path="", example=""):
        if local:
            model_checkpoint = ""
        else:
            model_checkpoint = path
        mask_filler = pipeline(
            "fill-mask",
            model=model_checkpoint,
        )
        print(mask_filler(example))

if __name__ == "__main__":
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    1_LOADING DATASET
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    mlm = MaskedLM()
    mlm.load_dataset()
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    2_EXPLORING DATASET, 
      _CREATING MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    print("-"*50, "Exploring information of Supporting", "-"*50)
    mlm.load_support()
    print("-"*50, "Exploring information of Supporting", "-"*50)
    print("-"*50, f"Information of {mlm.model_checkpoint}", "-"*50)
    mlm.create_model()
    print("-"*50, f"Information of {mlm.model_checkpoint}", "-"*50)
    print("-"*50, "Exploring some information of Model", "-"*50)
    mlm.explore_infoModel()
    print("-"*50, "Exploring some information of Model", "-"*50)
    print("Example[0] (text) in dataset: ", mlm.get_feature_items(set="train", index=0, feature="text")[:100] + "...")
    print("Example[0] (label) in dataset: ", mlm.get_feature_items(set="train", index=0, feature="label"))
    line1, line2 = mlm.get_pair_items(set="train", index=1, feature1="text", feature2="label")
    print("--> Inp of Example[1]: ", line1[:20] + "...")
    print("--> Out of Example[1]: ", line2[:20]+ "...")

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    3_PRE-PROCESSING DATASET, 
      _COMPUTE METRICS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    tokens, word_ids = mlm.get_tokenizer(set="train", index=0, feature="text")
    print("Tokens List of Example 0: ",tokens)
    print("Word IDs List of Example 0: ",word_ids)
    mlm.map_tokenize_dataset()
    mlm.map_chunk_dataset()
    mlm.dataset_split()

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    4_SELECTION HYPERPARMETERS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    mlm.create_argumentTrainer(push_to_hub=True, hub_model_id="Chessmen/"+"fine_tune_" + mlm.model_checkpoint)
    mlm.call_train(save_local=True,push_to_hub=True)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        5_USE PRE-TRAINED MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    mlm.call_pipeline(path="Chessmen/fine_tuned_distilbert-base-uncased",example= "This is a great [MASK].")

