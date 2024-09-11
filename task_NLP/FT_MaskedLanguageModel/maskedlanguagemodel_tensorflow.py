from sched import scheduler
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TFAutoModelForMaskedLM,
    PushToHubCallback,
    create_optimizer,
    pipeline
)
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf
import evaluate
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MaskedLM(object):
    def __init__(self):
        self.model = None
        self.data_collator = None
        self.tokenizer = None
        self.raw_data = None
        self.model_checkpoint = None
        self.tokenized_dataset = None
        self.chunk_size = 128
        self.chunks_dataset = None
        self.split_dataset = None
        self.optimizer = None
        self.train_dataset = None
        self.test_dataset = None
        self.scheduler = None

    def load_dataset(self, name="imdb"):
        self.raw_data = load_dataset(name)
        print("Name of dataset: ", name)
        print(self.raw_data)

    def load_support(self, mlm_probability=0.15):
        self.model_checkpoint = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=mlm_probability)
        print("Name of model checkpoint: " + self.model_checkpoint)
        print("Tokenizer Fast: ", self.tokenizer.is_fast)
        print("Symbol of masked word after tokenizer: ", self.tokenizer.mask_token)
        print("Model max length tokenizer: ", self.tokenizer.model_max_length)

    def explore_infoModel(self, k=5):
        model = TFAutoModelForMaskedLM.from_pretrained(self.model_checkpoint)
        model_parameters = model.num_parameters() / 1_000_000
        print(f">>> Number of parameters of {self.model_checkpoint}: {round(model_parameters)}M")

        example = "This is a great [MASK]."
        print("\n")
        print(">>> Example: ", example)
        inputs = self.tokenizer(example, return_tensors="tf")
        token_logits = model(**inputs).logits
        print(f"{'Number of tokens: ':<{30}}{ len(inputs.tokens())}")
        print(f"{'Tokens prepare for training: ':<{30}}{ inputs.tokens()}")
        print(f"{'IDs of tokens: ':<{30}}{ inputs.input_ids}")
        print(f"{'Maked IDs of Model: ':<{30}}{self.tokenizer.mask_token_id}")
        print(f"{'Logits of example: ':<{30}}{token_logits}")
        print(f"{'Shape of logits: ':<{30}}{token_logits.shape}")

        '''Find POSITION of [MASK] => EXTRACT LOGIT'''
        mask_token_index = tf.where(inputs.input_ids == self.tokenizer.mask_token_id)[0,1]
        print(f"{'Position of masked token index: ':<{30}}{mask_token_index}")

        '''Find LOGIT of token in VOCAB suitable for [MASK]'''
        mask_token_logits = token_logits[0, mask_token_index, :]
        print(f"{'Logit of tokens in Vocab for [MASK]: ':<{30}}{mask_token_logits}")

        '''Choose TOP CANDIDATES for [MASK] with highest logits => TOP LOGITS + POSITION of token suitable for [MASK] in VOCAB'''
        top_k_values = tf.math.top_k(mask_token_logits, k=k).values.numpy().tolist()
        print(f"{'Top value of suitable token in Vocab: ':<{30}}{top_k_values }")
        top_k_tokens = tf.math.top_k(mask_token_logits, k=k).indices.numpy().tolist()
        print(f"{'Position of suitable token in Vocab: ':<{30}}{top_k_tokens}")\
        
        '''Show TOP CANDIDATES'''
        for token in top_k_tokens:
            print(">>> ", example.replace(self.tokenizer.mask_token, self.tokenizer.decode([token])))

    def get_feature_items(self, set="train", index=0, feature="text"):
        return None if self.raw_data[set][index][feature] is None or self.raw_data[set][index][feature] == 0 else self.raw_data[set][index][feature]
    
    def get_pair_items(self, set="train", index=0, feature1="text", feature2="label"):
        feature1 = self.get_feature_items(set, index, feature1)
        feature2 = self.get_feature_items(set, index, feature2)
        if feature2 is not None:
            line1= ""
            line2= ""
            for word, label in zip(feature1, feature2):
                line1+= str(word)
                line2+= str(label)
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
        self.tokenized_dataset = self.raw_data.map(self.tokenizer_dataset, batched=True, remove_columns=["text","label"])
        print("Done mapping")
        print("Tokenized dataset: ", self.tokenized_dataset)

    def group_text_chunk(self, example):
        '''Group all of text'''
        concatenate_example = {k: sum(example[k],[]) for k in example.keys()}

        '''Compute the length of all'''
        total_length = len(concatenate_example["input_ids"])

        '''Final length for chunk size'''
        total_length = (total_length // self.chunk_size) * self.chunk_size

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

    def dataset_split(self, test_size=0.2,):
        self.split_dataset = self.chunks_dataset["train"].train_test_split(
            test_size=test_size, seed=42
        )
        print("Preparing dataset: ", self.split_dataset)

    def convert_tf_type(self,batch_size=64):
        print("Convert dataset to TF type")
        self.train_dataset = self.split_dataset["train"].to_tf_dataset(
            columns=['attention_mask', 'input_ids', 'labels'],
            collate_fn=self.data_collator,
            shuffle=True,
            batch_size=batch_size,
        )
        self.test_dataset = self.split_dataset["test"].to_tf_dataset(
            columns=['attention_mask', 'input_ids', 'labels'],
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=batch_size,
        )
        print("Done converting dataset to TF type")
    def create_model(self):
        print("Start creating model")
        self.model = TFAutoModelForMaskedLM.from_pretrained(self.model_checkpoint)
        print(self.model)

    def create_hyperparameter(self, learning_rate=2e-5, weight_decay=0.01,
                              num_warmup_steps=1_000,num_train_epochs=20):
        #tf.keras.mixed_precision.set_global_policy("mixed_float16")
        self.epochs = num_train_epochs
        num_train_steps = len(self.train_dataset) * self.epoch
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
            output_dir = output_dir + self.model_checkpoint
            callback = PushToHubCallback(output_dir=output_dir , save_strategy=save_strategy,
                                                            tokenizer=self.tokenizer, hub_model_id=hub_model_id, checkpoint=checkpoint)
        else:
            callback = None
        self.model.compile(optimizer=self.optimizer)

        eval_result1 = self.model.evaluate(self.test_dataset)
        print("Perplexity before of training: ", math.exp(eval_result1))

        self.model.fit(
            self.train_dataset,
            validation_data=self.test_dataset,
            callbacks=[callback],
            epochs=self.epochs,
        )
        print("Done training")
        print("Done pushing push to hub")

        eval_result2 =  self.model.evaluate(self.test_dataset)
        print("Perplexity after of training: ", math.exp(eval_result2))

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
    '''
        1_LOADING DATASET
    '''
    mlm = MaskedLM()
    mlm.load_dataset()
    print("-"*50, "Exploring information of Supporting", "-"*50)
    mlm.load_support()
    print("-"*50, "Exploring information of Supporting", "-"*50)
    '''
        2_EXPLORING DATASET, MODEL
    '''
    print("-"*50, "Exploring some information of Model", "-"*50)
    mlm.explore_infoModel()
    print("-"*50, "Exploring some information of Model", "-"*50)
    print("Example[0] (text) in dataset: ", mlm.get_feature_items(set="train", index=0, feature="text")[:100] + "...")
    print("Example[0] (label) in dataset: ", mlm.get_feature_items(set="train", index=0, feature="label"))
    line1, line2 = mlm.get_pair_items(set="train", index=1, feature1="text", feature2="label")
    print("--> Inp of Example[1]: ", line1[:20] + "...")
    print("--> Out of Example[1]: ", line2[:20]+ "...")
    '''
        3_PRE-PROCESSING DATASET, COMPUTE METRICS
    '''
    tokens, word_ids = mlm.get_tokenizer(set="train", index=0, feature="text")
    print("Tokens List of Example 0: ",tokens)
    print("Word IDs List of Example 0: ",word_ids)
    mlm.map_tokenize_dataset()
    mlm.map_chunk_dataset()
    mlm.dataset_split()
    mlm.convert_tf_type()
    '''
        4_INITIALIZATION MODEL
    '''
    print("-"*50, f"Information of {mlm.model_checkpoint}", "-"*50)
    mlm.create_model()
    print("-"*50, f"Information of {mlm.model_checkpoint}", "-"*50)
    '''
        5_SELECTION HYPERPARMETERS
    '''
    mlm.create_hyperparameter()
    mlm.call_train(save_local=True,push_to_hub=True, hub_model_id="Chessmen/TF_fine_tune_" + mlm.model_checkpoint)
    '''
        6_USE PRE-TRAINED MODEL
    '''
    mlm.call_pipeline(path="Chessmen/fine_tuned_distilbert-base-uncased",example= "This is a great [MASK].")