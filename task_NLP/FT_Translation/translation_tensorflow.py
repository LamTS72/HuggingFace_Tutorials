from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TFAutoModelForSeq2SeqLM,
    pipeline,
    create_optimizer
)
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf
import evaluate
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Translation(object):
    def __init__(self):
        self.model = None
        self.data_collator = None
        self.raw_data = None
        self.model_checkpoint = None
        self.tokenizer = None
        self.tokenized_dataset = None
        self.split_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.metric = None
        self.optimizer = None
        self.scheduler = None
        self.max_input_length  = None
        self.max_target_length = None
        self.other_name = None

    def load_dataset(self, name="kde4", lang1="en", lang2="fr"):
        self.raw_data = load_dataset(name, lang1=lang1, lang2=lang2)
        print("Name of dataset: ", name)
        print(self.raw_data)

    def load_support(self, name="Helsinki-NLP/opus-mt-en-fr"):
        self.model_checkpoint = name
        self.other_name = "marian-finetuned-kde4-en-to-fr"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, return_tensor="tf")
        print("Name of model checkpoint: ", self.model_checkpoint)
        print("Tokenizer Fast: ", self.tokenizer.is_fast)
        print("Model max length tokenizer: ", self.tokenizer.model_max_length) 

    def create_model(self):
        print("Start creating model")
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint, from_pt=True)
        print(self.model)

    def create_collator(self):
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, return_tensors="tf")

    def dataset_split(self, set="train", test_size=0.2):
        self.split_dataset = self.raw_data[set].train_test_split(test_size=test_size, seed=42)
        print(self.split_dataset)

    def get_feature_items(self, set="train", index=0, feature="translation"):
        return self.split_dataset[set][index][feature]

    def get_pair_feature(self, set="train", index=0, feature="translation", attribute1="en", attribute2="fr"):
        line1 = self.get_feature_items(set, index, feature)[attribute1]
        line2 = self.get_feature_items(set, index, feature)[attribute2]
        return line1, line2
    
    def tokenize_dataset(self, example):
        inputs = [ex["en"] for ex in example["translation"]]
        targets = [ex["fr"] for ex in example["translation"]]

        '''Tokenize for INPUTS'''
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)

        '''Tokenize for TARGETS in context'''
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length,truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def map_tokenize_dataset(self):
        self.tokenized_dataset = self.split_dataset.map(
            self.tokenize_dataset, 
            batched=True,
            remove_columns=self.split_dataset["train"].column_names
        )
        print("Tokenized dataset for training: ", self.tokenized_dataset)

    def convert_tf_type(self, batch_size=64):
        self.train_dataset = self.tokenized_dataset["train"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=self.data_collator,
            shuffle=True,
            batch_size=batch_size,
        )
        self.test_dataset = self.tokenized_dataset["test"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=batch_size,
        )

    def get_tokenizer(self, set="train", index=0):
        example = self.data_collator([self.tokenized_dataset[set][index]])
        print("Dictionary keys in an example: ", example.keys())
        print("IDs in an example: ", example["input_ids"])
        print("Labels in an example: ", example["labels"])

    def create_metric(self, name="sacrebleu"):
        self.metric = evaluate.load(name)

    def compute_metric(self, size=300):
        all_predictions = []
        all_labels = []
        batch_size = 16
        eval_dataset = self.tokenized_dataset["test"].shuffle().select(range(size))
        test_dataset = eval_dataset.to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=batch_size,
        )
        for batch in test_dataset:
            '''Get PREDICTIONS'''
            predictions = self.model.generate(
                 input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            '''Decode LOGIT of prediction to TEXT'''            
            decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

            '''Bacause LABEL use [-100] for padding NOT SAME MODEL pad_token_id => convert to [PAD]'''
            labels = batch["labels"].numpy()
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            '''Handle post processing for strings => strip() to remove space front and rear'''
            decoded_predictions = [prediction.strip() for prediction in decoded_predictions]
            decoded_labels = [label.strip() for label in decoded_labels]

            ''''Add to list'''
            all_predictions.extend(decoded_predictions)
            all_labels.extend(decoded_labels)

        result = self.metric.compute(predictions=all_predictions, references=all_labels, use_stemmer=True)
        return {"bleu": result}

    def  create_hyperparameter(self, learning_rate=2e-5, weight_decay=0.01,
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
        print("Basis evaluation before of training: ",self.compute_metric(size=100))

        self.model.fit(
            self.train_dataset,
            validation_data=self.test_dataset,
            callbacks=[callback],
            epochs=self.epochs,
        )
        print("Done training")
        print("Done pushing push to hub")

        print("Basis evaluation after of training: ",self.compute_metric(size=100))

    def call_pipeline(self, local=False, path="", example=""):
        if local:
            model_checkpoint = ""
        else:
            model_checkpoint = path
        translator = pipeline(
            "translation",
            model=model_checkpoint,
        )
        print(translator(example))

if __name__ == "__main__":
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    1_LOADING DATASET
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    translator = Translation()
    translator.load_dataset()

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    2_EXPLORING DATASET, 
      _CREATING MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    translator.load_support()
    translator.create_model()
    translator.create_collator()
    translator.dataset_split()
    print(translator.get_feature_items(set="train", index=0, feature="translation"))
    line1, line2 =translator.get_pair_feature(set="train", index=0, feature="translation", attribute1="en", attribute2="fr")
    print("--> Inp of Example[1]: ", line1)
    print("--> Out of Example[1]: ", line2)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    3_PRE-PROCESSING DATASET, 
      _COMPUTE METRICS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    translator.map_tokenize_dataset()
    translator.get_tokenizer()
    translator.convert_tf_type(batch_size=32)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    4_SELECTION HYPERPARMETERS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    translator.create_hyperparameter(num_train_epochs=5)
    translator.call_train(save_local=True,push_to_hub=True, hub_model_id="Chessmen/TF_fine_tune_" + translator.other_name)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        5_USE PRE-TRAINED MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    translator.call_pipeline(path="Chessmen/TF_fine_tuned_marian-finetuned-kde4-en-to-fr",example= "Default to expanded threads")