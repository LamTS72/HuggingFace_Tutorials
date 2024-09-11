# -*- coding: utf-8 -*-
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
# from huggingface_hub import login()
import evaluate
import numpy as np
from torch.utils.data import DataLoader
from configparser import ConfigParser

# config_object = ConfigParser()
# config_object.read("./config.ini")
# user = config_object["HUGGINGFACE"]
# print(user["key"])

class TokenClassifier():
    def __init__(self):
        self.model = None
        self.labelName = None
        self.metric = None
        self.data_collator = None
        self.tokenizer = None
        self.raw_datasets = None
        self.model_checkpoint = None
        self.args = None
        self.tokenized_dataset = None

    def load_dataset(self):
        self.raw_datasets = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")
        print(self.raw_datasets)
    
    def load_support(self):
        self.model_checkpoint = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.metric = evaluate.load("seqeval")
    
    def get_feature_items(self, set="train",index=0, feature="tokens"):
        return self.raw_datasets[set][index][feature]

    def get_labelName_items(self, set="train", featureName="ner_tags"):
        self.labelName = self.raw_datasets[set].features[featureName].feature.names
        return self.labelName
    
    def get_pair_items(self, set="train", index=0, feature1="tokens", feature2="ner_tags"):
        feature1 = self.get_feature_items(set, index, feature1)
        feature2 = self.get_feature_items(set, index, feature2)
        line1 = ""
        line2 = ""
        for word, label in zip(feature1, feature2):
            full_label = self.labelName[label]
            max_length = max(len(word), len(full_label))
            line1 += str(word) + str(" " * (max_length - len(word) + 1))
            line2 += str(full_label) + str(" " * (max_length - len(full_label) + 1))

        return line1, line2

    def get_tokenizer(self, set="train",index=0, feature="tokens", is_split_into_words=True):
        print("Fast Token: ", self.tokenizer.is_fast)
        inputs = self.tokenizer(self.get_feature_items(set, index, feature), is_split_into_words=is_split_into_words)
        return inputs.tokens(), inputs.word_ids()
    
    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start with new word
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token 
                new_labels.append(-100)
            else:
                # Word same previous token
                label = labels[word_id]
                # If label is B-XXX, convert to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels
    
    def tokenize_and_align_labels(self, examples, feature1="tokens", feature2="ner_tags"):
        tokenized_inputs = self.tokenizer(
            examples[feature1], truncation=True, is_split_into_words=True
        )
        all_labels = examples[feature2]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    def map_tokenize_dataset(self, set="train"):
        print("Start of processing dataset")
        self.tokenized_dataset = self.raw_datasets.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=self.raw_datasets[set].column_names,
        )
        print("Done mapping")

    def compute_metrics(self, eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)

        # Remove special token and convert to labels
        true_labels = [[self.labelName[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.labelName[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    def create_model(self, id2label, label2id):
        print("Start creating model")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            id2label=id2label,
            label2id=label2id,
        )
        print("Number of labels in NER: ",self.model.config.num_labels)

    def convert_id_label(self):
        id2label = {i: label for i, label in enumerate(self.labelName)}
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id
    
    def create_argumentTrainer(self, output_dir="fine_tuned_", eval_strategy="epoch", logging_strategy="epoch",
                               save_strategy="epoch", learning_rate=2e-5,num_train_epochs=20, weight_decay=0.01, 
                               batch_size=8, push_to_hub=False, hub_model_id=""):
        self.args = TrainingArguments(
            use_cpu=True,
            output_dir=output_dir+self.model_checkpoint,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            logging_strategy=logging_strategy,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            #warmup_steps=0,
            #max_steps=num_training_steps,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size
        )
        
        print("Arguments ready for training")
        return self.args

    def call_train(self, model_path="pretrained_model_",set_train="train", set_val="validation", push_to_hub=False, save_local=False):
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.tokenized_dataset[set_train],
            eval_dataset=self.tokenized_dataset[set_val],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        print("Start training")
        trainer.train()
        print("Done training")
        if save_local:
            trainer.save_model(model_path + self.model_checkpoint)
            print("Done saving to local")
            # trainer.model.save_pretrained("model_pretrained")
        if push_to_hub:
            trainer.push_to_hub(commit_message="Training complete")
            print("Done pushing push to hub")

    def call_pipeline(self, local=False, path="fine_tuned_model", example=""):
        if local:
            model_checkpoint = "token_classify"
        else:
            model_checkpoint = path
            
        token_classifier = pipeline(
            "token-classification",
            model=model_checkpoint,
            aggregation_strategy="simple",
        )
        print(token_classifier(example))


if __name__ == "__main__":
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    1_LOADING DATASET
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    ner = TokenClassifier()
    ner.load_dataset()

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    2_EXPLORING DATASET, 
      _CREATING MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    print("Example 0(tokens) in dataset: ",ner.get_feature_items(set="train", index=0, feature="tokens"))
    print("Example 0(ner_tags) in dataset: ",ner.get_feature_items(set="train", index=0, feature="ner_tags"))
    print("List labels name in NER: ",ner.get_labelName_items(set="train", featureName="ner_tags"))
    id2label, label2id = ner.convert_id_label()
    ner.load_support()
    ner.create_model(id2label, label2id)
  
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    3_PRE-PROCESSING DATASET, 
      _COMPUTE METRICS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    print("Dict of IDs into labels: ", id2label)
    print("Pair of Example 0")
    line1, line2 = ner.get_pair_items(set="train", index=0, feature1="tokens", feature2="ner_tags")
    print("--> Inp of Example 0: ", line1)
    print("--> Out of Example 0: ", line2)
    tokens, word_ids= ner.get_tokenizer(set="train", index=0, feature="tokens", is_split_into_words=True)
    print("Tokens List of Example 0: ",tokens)
    print("Word IDs List of Example 0: ",word_ids)
    ner.map_tokenize_dataset(set="train")

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    4_SELECTION HYPERPARMETERS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    ner.create_argumentTrainer(batch_size=16, push_to_hub=True, hub_model_id="Chessmen/"+"fine_tune_" + ner.model_checkpoint)
    ner.call_train(push_to_hub=True)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        5_USE PRE-TRAINED MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    ner.call_pipeline(example="My name is Sylvain and I work at Hugging Face in Brooklyn.")