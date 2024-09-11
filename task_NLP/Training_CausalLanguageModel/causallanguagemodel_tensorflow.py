from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import(
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoConfig,
    TFGPT2LMHeadModel,
    create_optimizer,
    pipeline
)
import evaluate
import numpy as np
import tensorflow as tf
import math

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

    def load_dataset(self, name1="huggingface-course/codeparrot-ds-train", name2= "huggingface-course/codeparrot-ds-valid"):
        train_dataset = load_dataset(name1, split="train")
        val_dataset = load_dataset(name2, split="validation")
        self.raw_dataset = DatasetDict(
            {
                "train":  train_dataset,#.shuffle().select(range(20000)),
                "validation": val_dataset#.shuffle().select(range(200))
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
        self.model = TFGPT2LMHeadModel(config)
        self.model(self.model.dummy_inputs)
        print("Information of model : ",self.summary())

    def create_collator(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False, return_tensors="tf")

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

    def convert_tf_type(self, batch_size=64):
        self.train_dataset = self.tokenized_dataset["train"].to_tf_dataset(
                columns=["input_ids", "attention_mask", "labels"],
                collate_fn=self.data_collator,
                shuffle=True,
                batch_size=batch_size,
            )
        self.val_dataset = self.tokenized_dataset["validation"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=batch_size,
        )            

    def  create_hyperparameter(self, learning_rate=5e-5, weight_decay=0.01,
                              num_warmup_steps=1_000, num_train_epochs=20):
        self.epochs = num_train_epochs
        num_train_steps = len(self.train_dataset)*self.epochs
        self.optimizer, self.scheduler = create_optimizer(
            init_lr=learning_rate,
            num_warmup_steps=num_warmup_steps,
            num_train_steps=num_train_steps,
            weight_decay_rate=weight_decay,
        )
        print("Optimizer ready for training")
        return self.optimizer, self.scheduler

    def call_train(self, output_dir="TF_fine_tuned_", save_strategy="epoch",push_to_hub=False,
                    save_local=False, hub_model_id="", checkpoint=False):
        if push_to_hub:
            output_dir = output_dir + self.other_name
            callback = PushToHubCallback(output_dir=output_dir , save_strategy=save_strategy,
                                                            tokenizer=self.tokenizer, hub_model_id=hub_model_id, checkpoint=checkpoint)
        else:
            callback = None
        self.model.compile(optimizer=self.optimizer)
        #tf.keras.mixed_precision.set_global_policy("mixed_float16")
        eval_result1 = self.model.evaluate(self.val_dataset)
        print("Perplexity before of training: ", math.exp(eval_result1))

        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            callbacks=[callback],
            epochs=self.epochs,
        )
        print("Done training")
        print("Done pushing push to hub")

        eval_result2 = self.model.evaluate(self.val_dataset)
        print("Perplexity after of training: ", math.exp(eval_result2))

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

    clm.create_hyperparameter()
    clm.call_train(save_local=True,push_to_hub=True, hub_model_id="Chessmen/TF_fine_tune_" + clm.other_name)

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
