# -*- coding: utf-8 -*-
from datasets import load_dataset
from sklearn import metrics
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TFAutoModelForTokenClassification,
    PushToHubCallback,
    create_optimizer,
    pipeline
)
from datasets import load_dataset
import evaluate
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TokenClassifier():
    def __init__(self):
        self.model = None
        self.labelName = None
        self.metric = None
        self.data_collator = None
        self.tokenizer = None
        self.raw_datasets = None
        self.model_checkpoint = None
        self.tf_train_dataset = None
        self.tf_val_dataset = None
        self.optimizer = None
        pass

    def load_dataset(self):
        self.raw_datasets = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")
        print(self.raw_datasets)
    
    def load_model(self):
        self.model_checkpoint = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, return_tensors="tf")
        self.metric = evaluate.load("seqeval")
    
    def create_model(self, id2label, label2id):
        print("Start creating model")
        self.model = TFAutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            id2label=id2label,
            label2id=label2id
        )
        print("Number of labels in NER: ", self.model.config.num_labels)

    def convert_id_label(self):
        id2label = {i: label for i, label in enumerate(self.labelName)}
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id
    
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
    
    def get_tokenizer(self, set="train", index=0, feature="tokens", is_split_into_words=True):
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


    def tokenize_and_align_labels(self, examples,  feature1="tokens", feature2="ner_tags"):
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

    def dataset_loader_tf(self):
        print("Convert dataset to TF type")
        self.tf_train_dataset = self.tokenized_dataset["train"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "token_type_ids","labels"],
            #label_cols=["labels"],
            collate_fn=self.data_collator,
            shuffle=True,
            batch_size=8,
        )

        self.tf_val_dataset = self.tokenized_dataset["validation"].to_tf_dataset(
            columns=["attention_mask", "input_ids","token_type_ids","labels"],
            #label_cols=["labels"],
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=8,
        )
        print("Done converting TF")
    
    def compute_metrics(self, eval_prediction):
        all_predictions = []
        all_labels = []
        for batch in eval_prediction:
            logits = self.model.predict(batch)["logits"]
            labels = batch["labels"]
            predictions = np.argmax(logits, axis=-1)
            for prediction, label in zip(predictions, labels):
                for predicted_idx, label_idx in zip(prediction, label):
                    if label_idx == -100:
                        continue
                    all_predictions.append(self.labelName[predicted_idx])
                    all_labels.append(self.labelName[label_idx])

        self.metric.compute(predictions=[all_predictions], references=[all_labels])

    def create_hyperparameter(self, learning_rate=2e-5,num_train_epochs=1, weight_decay=0.01):
        #tf.keras.mixed_precision.set_global_policy("mixed_float16")
        num_train_steps = len(self.tf_train_dataset) * num_train_epochs
        self.optimizer, _= create_optimizer(
            init_lr=learning_rate,
            num_warmup_steps=0,
            num_train_steps=num_train_steps,
            weight_decay_rate=weight_decay,
        )
        print("Optimizer ready for training")
        return self.optimizer

    def call_train(self, num_train_epochs=1, output_dir="fine_tuned_model", save_strategy="epoch",push_to_hub=False,
                    save_local=False, hub_model_id="Chessmen/test"):
        if push_to_hub:
            callback = PushToHubCallback(output_dir=output_dir, save_strategy=save_strategy,
                                         tokenizer=self.tokenizer, hub_model_id=hub_model_id, checkpoint=True)
        else:
            callback = None
        self.model.compile(optimizer=self.optimizer)
        self.model.fit(
            self.tf_train_dataset,
            validation_data=self.tf_val_dataset,
            callbacks=[callback],
            epochs=num_train_epochs,
        )
        print("Done training")
        print("Done pushing push to hub")

    def call_pipeline(self, local=True, path_url="fine_tuned_model", example=""):
        if local:
            model_checkpoint = path_url
        else:
            model_checkpoint = "token_classify"
            
        token_classifier = pipeline(
            "token-classification",
            model=model_checkpoint,
            aggregation_strategy="simple",
        )
        print(token_classifier(example))

if __name__=="__main__":
    '''
        1_LOADING DATASET
    '''
    ner = TokenClassifier()
    ner.load_dataset()
    ner.load_model()
    '''
        2_EXPLORING DATASET
    '''
    print("Example 0(tokens) in dataset: ",ner.get_feature_items(set="train", index=0, feature="tokens"))
    print("Example 0(ner_tags) in dataset: ",ner.get_feature_items(set="train", index=0, feature="ner_tags"))
    '''
        3_PRE-PROCESSING DATASET, COMPUTE METRICS
    '''
    print("List labels name in NER: ",ner.get_labelName_items(set="train", featureName="ner_tags"))
    id2label, label2id = ner.convert_id_label()
    print("Dict of IDs into labels: ", id2label)
    print("Pair of Example 0")
    line1, line2 = ner.get_pair_items(set="train", index=0, feature1="tokens", feature2="ner_tags")
    print("--> Inp of Example 0: ", line1)
    print("--> Out of Example 0: ", line2)
    tokens, word_ids= ner.get_tokenizer(set="train", index=0, feature="tokens", is_split_into_words=True)
    print("Tokens List of Example 0: ",tokens)
    print("Word IDs List of Example 0: ",word_ids)
    ner.map_tokenize_dataset(set="train")
    # print("(ner_tags) in dataset: ",ner.get_feature_items(set="train", index=0, feature="ner_tags"))
    # batch = ner.data_collator([ner.tokenized_dataset["train"][i] for i in range(2)])
    # print(batch["labels"])
    ner.dataset_loader_tf()
    '''
        4_INITIALIZATION MODEL
    '''
    ner.create_model(id2label, label2id)
    '''
        5_SELECTION HYPERPARAMETERS
    '''
    ner.create_hyperparameter()
    ner.call_train(save_local=True ,push_to_hub=True)
    print(ner.compute_metrics(ner.tf_val_dataset))
    '''
        6_USE PRE-TRAINED MODEL
    '''
    ner.call_pipeline(example="My name is Sylvain and I work at Hugging Face in Brooklyn.")