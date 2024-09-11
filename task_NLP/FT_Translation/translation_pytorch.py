
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline
)
import evaluate
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Translation(object):
    def __init__(self):
        self.model = None
        self.data_collator = None
        self.raw_data = None
        self.model_checkpoint = None
        self.tokenized_dataset = None
        self.split_dataset = None
        self.max_input_length = 128
        self.max_target_length = 128
        self.args = None
        self.metric = None
        self.other_name = None

    def load_dataset(self, name="kde4", lang1="en", lang2="fr"):
        self.raw_data = load_dataset(name, lang1=lang1, lang2=lang2)
        print("Name of dataset: ", name)
        print(self.raw_data)

    def load_support(self, name="Helsinki-NLP/opus-mt-en-fr") :
        self.model_checkpoint = name
        self.other_name = "marian-finetuned-kde4-en-to-fr"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        print("Name of model checkpoint: ", self.model_checkpoint)
        print("Tokenizer Fast: ", self.tokenizer.is_fast)
        print("Model max length tokenizer: ", self.tokenizer.model_max_length)
    
    def create_model(self):
        print("Start creating model")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        print(self.model)
    
    def create_collator(self):
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

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
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def map_tokenize_dataset(self):
        self.tokenized_dataset = self.split_dataset.map(
            self.tokenize_dataset, 
            batched=True,
            remove_columns=self.split_dataset["train"].column_names
        )
        print("Tokenized dataset for training: ", self.tokenized_dataset)
        

    def get_tokenizer(self, set="train", index=0):
        example = self.data_collator([self.tokenized_dataset[set][index]])
        print("Dictionary keys in an example: ", example.keys())
        print("IDs in an example: ", example["input_ids"])
        print("Labels in an example: ", example["labels"])

    def create_metric(self, name="sacrebleu"):
        self.metric = evaluate.load(name)

    def compute_metrics(self, eval_prediction):
        predictions, labels = eval_prediction
        '''In some cases, the model returns more than one prediction => GET FIRST prediction'''
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        '''Decode LOGIT of prediction to TEXT'''
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        '''Bacause LABEL use [-100] for padding NOT SAME MODEL pad_token_id => convert to [PAD]'''
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        '''Handle post processing for strings => strip() to remove space front and rear'''
        decoded_predictions = [prediction.strip() for prediction in decoded_predictions]
        decoded_labels = [label.strip() for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_predictions, labels=decoded_labels)
        return {"bleu": result}
    
    def create_argumentTrainer(self, output_dir="fine_tuned_", eval_strategy="epoch", logging_strategy="epoch",
                               learning_rate=5.6e-5, num_train_epochs=20, weight_decay=0.01, batch_size=64,
                               save_strategy="epoch", push_to_hub=False, hub_model_id="", fp16=True,save_total_limit=3,
                               predict_with_generate=True):
        self.args = Seq2SeqTrainingArguments(
            use_cpu=True,
            output_dir=output_dir+self.other_name,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            logging_strategy=logging_strategy,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            save_total_limit=save_total_limit,
            predict_with_generate=predict_with_generate,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size
        )
        print("Arguments ready for training")
        return self.args
    
    def call_train(self, model_path="pretrained_model_", set_train="train", set_val="test", push_to_hub=False, save_local=False):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.tokenized_dataset[set_train],
            eval_dataset=self.tokenized_dataset[set_val],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        evaluate1 =  trainer.evaluate(max_length=self.max_target_length)
        print("Basis evaluation before of training: ",evaluate1)

        print("Start training")
        trainer.train()
        print("Done training")

        evaluate2 = trainer.evaluate(max_length=self.max_target_length)
        print("Basis evaluation after of training: ", evaluate2)
        
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

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    4_SELECTION HYPERPARMETERS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    translator.create_argumentTrainer(push_to_hub=True, hub_model_id="Chessmen/"+"fine_tune_" + translator.other_name)
    translator.call_train(save_local=True,push_to_hub=True)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        5_USE PRE-TRAINED MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    translator.call_pipeline(path="Chessmen/fine_tuned_marian-finetuned-kde4-en-to-fr",example= "Default to expanded threads")