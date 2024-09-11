from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import(
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoConfig,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    pipeline
)
import evaluate
import numpy as np
import torch
import math

device = torch.device("cude" if torch.cuda.is_available() else "cpu")
print(device)

class CausalLM(object):
    def __init__(self):
        self.model = None
        self.data_collator = None
        self.raw_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.model_checkpoint = None
        self.tokenizer = None
        self.tokenized_dataset = None
        self.context_length = 128
        self.args = None

    def load_dataset(self, name1="huggingface-course/codeparrot-ds-train", name2= "huggingface-course/codeparrot-ds-valid"):
        train_dataset = load_dataset(name1, split="train")
        val_dataset = load_dataset(name2, split="validation")
        self.raw_dataset = DatasetDict(
            {
                "train":  train_dataset.shuffle().select(range(500000)),
                "validation": val_dataset.shuffle().select(range(500))
            }
        )
        self.other_name = name1.split("/")[-1].split("-")[0]
        print("Name of dataset: ", self.other_name)
        print(self.raw_dataset)
    
    def load_support(self, name= "huggingface-course/code-search-net-tokenizer"):
        self.model_checkpoint = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        print("Name of model checkpoint: " + self.model_checkpoint)
        print("Tokenizer Fast: ", self.tokenizer.is_fast)
        print("Model max length tokenizer: ", self.tokenizer.model_max_length)

    def create_model(self):
        '''Do not use AutoModelFor* bacause CREATE NEW MODEL'''
        print("Start creating model")
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(self.tokenizer),
            n_ctx=self.context_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        self.model = GPT2LMHeadModel(config)
        model_parameters = self.model.num_parameters() / 1_000_000
        print(f">>> Number of parameters of {self.model_checkpoint}: {round(model_parameters)}M")

    def create_collator(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def get_example(self, set="train", index=0, length_text=200):
        for key in self.raw_dataset[set][index]:
            print(f"{key.upper()}: {self.raw_dataset[set][index][key][:length_text]}")

    def get_feature_items(self, set="train", index=0, feature="content"):
        print("Number of line in example: ", len(self.raw_dataset[set][index][feature].split()))
        return self.raw_dataset[set][index][feature]
    
    def get_tokenizer(self, set="train", index=0, feature="content"):
        inputs = self.tokenizer(
            self.raw_dataset[set][index][feature],
            truncation=True,
            max_length=self.context_length,
            return_overflowing_tokens=True,
            return_length=True
        )
        return inputs.input_ids, inputs.length, inputs.overflow_to_sample_mapping
    
    def tokenize_dataset(self, example):
        inputs = self.tokenizer(
            example["content"],
            truncation=True,
            max_length=self.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        inputs_batch = []
        for length, input_ids in zip(inputs.length, inputs.input_ids):
            if length == self.context_length:
                inputs_batch.append(input_ids)
        return {"input_ids": inputs_batch}
    
    def map_tokenize_dataset(self, path=""):
        try:
            self.tokenized_dataset = load_from_disk(path)
        except:
            print("Not found")
            self.tokenized_dataset = None
    
        if self.tokenized_dataset is None:
            self.tokenized_dataset = self.raw_dataset.map(
                self.tokenize_dataset,
                batched=True,
                remove_columns=self.raw_dataset["train"].column_names
            )
            self.tokenized_dataset.save_to_disk(path)
        print("Tokenized dataset: ", self.tokenized_dataset)

    def create_argumentTrainer(self, output_dir="fine_tuned_", eval_strategy="no", logging_strategy="epoch",
                                            learning_rate=5.6e-5, num_train_epochs=20, weight_decay=0.01, batch_size=64,
                                            save_strategy="epoch", push_to_hub=False, hub_model_id="", fp16=True, 
                                            save_total_limit=3, predict_with_generate=True, evaluation_strategy="steps", eval_steps=5_000, 
                                            logging_steps=5_000, gradient_accumulation_steps=8, lr_scheduler_type="cosine",warmup_steps=1_000,
                                            save_steps=5_000):
        self.args = TrainingArguments(
            use_cpu=True,
            output_dir=output_dir+self.other_name,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            lr_scheduler_type=lr_scheduler_type,
            learning_rate=learning_rate,
            save_steps=save_steps,
            fp16=fp16,
            push_to_hub=True,
            hub_model_id=hub_model_id,
        )
        print("Arguments ready for training")
        return self.args  
    def call_train(self, model_path="pretrained_model_", set_train="train", set_val="validation", push_to_hub=False, save_local=False):
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.tokenized_dataset[set_train],
            eval_dataset=self.tokenized_dataset[set_val],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        
        eval_result1 = trainer.evaluate()
        print("Perplexity before of training: ", math.exp(eval_result1['eval_loss']))

        print("Start training")
        trainer.train()
        print("Done training")

        if save_local:
            trainer.save_model(model_path+self.model_checkpoint)
            print("Done saving to local")
  
        if push_to_hub:
            trainer.push_to_hub(commit_message="Training complete")
            print("Done pushing push to hub")    

        eval_result2 = trainer.evaluate()
        print("Perplexity after of training: ", math.exp(eval_result2['eval_loss']))

    def call_pipeline(self, local=False, path="", example="", num_return_sequences=1):
        if local:
            model_checkpoint = ""
        else:
            model_checkpoint = path
        text_generator = pipeline(
            "text-generation",
            model=model_checkpoint,
        )
        print(text_generator(example, num_return_sequences=num_return_sequences))

if __name__ == "__main__":
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    1_LOADING DATASET
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    print("-"*50, "Exploring information of Dataset", "-"*50)
    clm = CausalLM()
    clm.load_dataset()
    print("-"*50, "Exploring information of Dataset", "-"*50)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    2_EXPLORING DATASET, 
      _CREATING MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    print("-"*50, "Exploring information of Supporting", "-"*50)
    clm.load_support()
    clm.create_model()
    clm.create_collator()
    print("-"*50, "Exploring information of Supporting", "-"*50)
    print("Example[0] in dataset: ")
    clm.get_example(set="train", index=0, length_text=200)
    print("\n")
    print("Content in example[0]: ", clm.get_feature_items(set="train", index=0,feature="content"))
    print("\n")
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    3_PRE-PROCESSING DATASET, 
      _COMPUTE METRICS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    input_ids, length, sample_mapping = clm.get_tokenizer(set="train", index=0, feature="content")
    print("Input IDs content of example[0]: ", input_ids)
    print("Length content of example[0]: ", length)
    print("Chunk sample mapping content of example[0]: ", sample_mapping)
    clm.map_tokenize_dataset(path="../ML_DL/HuggingFace_Tutorials/task_NLP/Training_CausalLanguageModel")

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    4_SELECTION HYPERPARMETERS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    clm.create_argumentTrainer(num_train_epochs=10, push_to_hub=True, hub_model_id="Chessmen/"+"fine_tune_" + clm.other_name)
    clm.call_train(save_local=False, push_to_hub=True)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        5_USE PRE-TRAINED MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    txt = """\
    # create some data
    x = np.random.randn(100)
    y = np.random.randn(100)

    # create scatter plot with x, y
    """
    clm.call_pipeline(path="Chessmen/fine_tuned_"+clm.other_name, example= txt, num_return_sequences=1)
