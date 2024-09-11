from json import load
from multiprocessing import context
from datasets import load_dataset
from transformers import(
    AutoTokenizer,
    TFAutoModelForQuestionAnswering,
    DefaultDataCollator,
    create_optimizer,
    pipeline
)
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf
import numpy as np
import evaluate
from tqdm.auto import tqdm
from collections import defaultdict

class QuestionAnswering(object):
    def __init__(self):
        self.model = None
        self.raw_dataset = None
        self.data_collator = None
        self.tokenizer = None
        self.tokenized_dataset = None
        self.model_checkpoint = None
        self.args = None
        self.metric = None
        self.max_length = 384
        self.stride = 128
        
    def load_dataset(self, name="squad"):
        self.model_checkpoint = name
        self.raw_dataset = load_dataset(name)
        print("Name of dataset: ", name)
        self.raw_dataset["train"].filter(lambda x: len(x["answers"]["text"]) != 1)
        print(self.raw_dataset)

    
    def load_support(self, name="bert-base-cased"):
        self.model_checkpoint = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        print("Name of model checkpoint: ", self.model_checkpoint)
        print("Tokenizer Fast: ", self.tokenizer.is_fast)
        print("Model max length tokenizer: ", self.tokenizer.model_max_length)       

    def create_model(self):
        print("Start creating model")
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(self.model_checkpoint)
        print(self.model)

    def create_collator(self):
        self.data_collator = DefaultDataCollator(return_tensors="tf")

    def create_metric(self, name="squad"):
       self.metric = evaluate.load(name)         

    def get_feature_items(self, set="train", index=0, feature="question"):
        return self.raw_dataset[set][index][feature]
    
    def get_triple_feature(self, set="train", index=0, feature1="question", feature2="context", feature3="answers"):
        line1 = self.get_feature_items(set, index, feature1)
        line2 = self.get_feature_items(set, index, feature2)
        line3 = self.get_feature_items(set, index, feature3)
        return line1, line2, line3
    
    def get_tokenizer(self, set="train", index=0, feature1="question", feature2="context"):
        question = self.raw_dataset[set][index][feature1]
        context = self.raw_dataset[set][index][feature2]
        inputs = self.tokenizer(
            question, 
            context,
            max_length=100,
            truncation="only_second",
            stride=50,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        print("Inputs keys: '", inputs.keys())
        print("Inputs example: ", inputs)
        print("List of input IDs: ", inputs.input_ids)
        print("List of tokens: ", inputs.tokens())
        print("\n")
        for i in inputs.input_ids:
            print(self.tokenizer.decode(i))

    def preprocess_training(self, example):
        question = [q.strip() for q in example["question"]]
        inputs = self.tokenizer(
            question,
            example["context"],
            max_length=self.max_length,
            stride=self.stride,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        answers = example["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids =inputs.sequence_ids(i)

            '''Find BEGIN position and END position of CONTEXT'''
            idx = 0
            '''BEGIN POSITION'''
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            '''END POSITION'''
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx  - 1
            '''Check ANSWER in CONTEXT in each pair of offset(start, end => (0, 0) if NONE '''
            if offset[context_start][0] > start_char or offset[context_end][0] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                '''EXIST' => traversal to right position'''
                idx = context_start
                '''START POSITION'''
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                '''END POSITION'''
                idx = context_end
                while idx >= context_start and offset[idx][0] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    def map_process_training(self):
        self.train_dataset = self.raw_dataset["train"].map(
            self.preprocess_training,
            batched=True,
            remove_columns=self.raw_dataset["train"].column_names
        )
        print("Length of original training dataset: ", len(self.raw_dataset["train"]))
        print( "Length of processing dataset: ", len(self.train_dataset))
              
    def process_validation(self, example):
        question = [q.strip() for q in example["question"]]
        inputs = self.tokenizer(
            question,
            example["context"],
            max_length=self.max_length,
            stride=self.stride,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        '''Example IDs store address of example IDs'''
        example_ids =[]

        for i in range(len(inputs.input_ids)):
            sample_idx = sample_mapping[i]
            example_ids.append(example["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(offset)
            ]
        
        inputs["example_id"] = example_ids
        return inputs

    def map_process_validation(self):
        self.val_dataset = self.raw_dataset["validation"].map(
            self.process_validation,
            batched=True,
            remove_columns=self.raw_dataset["validation"].column_names
        )
        print("Length of original validation dataset: ", len(self.raw_dataset["validation"]))
        print( "Length of processing dataset: ", len(self.val_dataset))

    def convert_tf_type(self, batch_size=64):
        self.train_dataset = self.train_dataset.to_tf_dataset(
            columns=[
                "input_ids",
                "start_positions",
                "end_positions",
                "attention_mask",
                "token_type_ids",
            ],
            collate_fn=self.data_collator,
            shuffle=True,
            batch_size=batch_size,
        )
        self.val_dataset = self.val_dataset.to_tf_dataset(
            columns=["input_ids", "attention_mask", "token_type_ids"],
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=batch_size,
        )
    def compute_metrics(self, start_logits, end_logits, features, examples):
        n_best = 20
        max_answer_length = 30
        example_to_features = defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)
        
        predicted_answers = []
        for example in tqdm(examples):
            example_id =  example["id"]
            context = example["context"]
            answers = []

            '''In ONE EXAMPLE have MORE THAN ONE FEATURES contain ANSWER => truncation text caused''' 
            for feature_idx in example_to_features[example_id]:
                start_logit = start_logits[feature_idx]
                end_logit = end_logits[feature_idx]
                offsets = features[feature_idx]["offset_mapping"]
                

                '''Get top 20 answers'''
                start_indexes = np.argsort(start_logit)[-1: -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1: -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        '''Skip if ANSWER not int context'''
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        '''Skip if ANSWER length < 0 or > max_answer_length'''
                        if (end_index < start_index or end_index - start_index + 1 > max_answer_length):
                            continue
                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index]
                        }
                        answers.append(answer)
            '''Choose highest answer score'''
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append({
                    "id":example_id,
                    "prediction_text": best_answer["text"]
                })
            else:
                predicted_answers.append({
                    "id": example_id,
                    "prediction_text": ""
                })
        theoretical_answers =[{"id": ex["id"], "answers": ex["answers"] } for ex in examples]
        return self.metric.compute(predictions=predicted_answers, references=theoretical_answers)
            
    def  create_hyperparameter(self, learning_rate=5.6e-5, weight_decay=0.01,
                              num_warmup_steps=0, num_train_epochs=20):
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
            output_dir = output_dir + self.model_checkpoint
            callback = PushToHubCallback(output_dir=output_dir , save_strategy=save_strategy,
                                                            tokenizer=self.tokenizer, hub_model_id=hub_model_id, checkpoint=checkpoint)
        else:
            callback = None
        self.model.compile(optimizer=self.optimizer)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        predictions, _, _ = self.model.predict(self.val_dataset)
        start_logits, end_logits = predictions
        print("Basis evaluation after of training: ", self.compute_metrics(start_logits, end_logits, self.val_dataset, self.raw_dataset["validation"]))

        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            callbacks=[callback],
            epochs=self.epochs,
        )
        print("Done training")
        print("Done pushing push to hub")     

        predictions, _, _ = self.model.predict(self.val_dataset)
        start_logits, end_logits = predictions
        print("Basis evaluation after of training: ", self.compute_metrics(start_logits, end_logits, self.val_dataset, self.raw_dataset["validation"]))

    def call_pipeline(self, local=False, path="", question="", context=""):
        if local:
            model_checkpoint = ""
        else:
            model_checkpoint = path
        qac = pipeline(
            "question-answering",
            model=model_checkpoint,
        )
        print("Answer: ",qac(question=question, context=context))

if __name__ == "__main__":
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    1_LOADING DATASET
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    print("-"*50, "Exploring information of Dataset", "-"*50)
    qac = QuestionAnswering()
    qac.load_dataset()
    print("-"*50, "Exploring information of Dataset", "-"*50)


    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    2_EXPLORING DATASET, 
      _CREATING MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    print("-"*50, "Exploring information of Supporting", "-"*50)
    qac.load_support()
    qac.create_model()
    qac.create_collator()
    qac.create_metric()
    print("-"*50, "Exploring information of Supporting", "-"*50)
    print("-"*50, "Exploring information of Supporting", "-"*50)
    print("Example[0] (question) in dataset: ", qac.get_feature_items( set="train", index=0, feature="question"))
    print("Example[0] (answers) in dataset: ", qac.get_feature_items( set="train", index=0, feature="answers"))
    print("\n")
    line1, line2, line3 = qac.get_triple_feature( set="train", index=0, feature1="question", feature2="context", feature3="answers")
    print("--> Question of Example[0]: ", line1)
    print("--> Context of Example[0]: ", line2)
    print("--> Answer of Example[0]: ", line3)
    print("\n")
    qac.get_tokenizer()
    print("\n")

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    3_PRE-PROCESSING DATASET, 
      _COMPUTE METRICS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    qac.map_process_training()
    qac.map_process_validation()

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    4_SELECTION HYPERPARMETERS
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    qac.create_hyperparameter(num_train_epochs=5, push_to_hub=True, hub_model_id="Chessmen/"+"TF_fine_tune_" + qac.model_checkpoint)
    qac.call_train(save_local=True,push_to_hub=True)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    5_USE PRE-TRAINED MODEL
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    context_q = """\
        Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration 
        between them. It's straightforward to train your models with one before loading them for inference with the other.
    """
    question = "Which deep learning libraries back 🤗 Transformers?"
    qac.call_pipeline(path="Chessmen/TF_fine_tuned_"+ qac.model_checkpoint, question=question, context=context_q)
