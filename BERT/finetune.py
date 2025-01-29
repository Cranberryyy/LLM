import sys
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification
from transformers import BertForQuestionAnswering
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
sys.path.append('./model_saved')


from transformers import DistilBertTokenizerFast

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


# Preprocess function to tokenize and align answers
def preprocess_function(examples):
    # Tokenize the inputs (question and context)
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map the position of the start and end of the answer to the tokenized input
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples





def main():
    squad_dataset = load_dataset("squad_v2")
    tokenized_train_dataset = squad_dataset["train"].map(preprocess_function, batched=True, remove_columns=squad_dataset["train"].column_names)
    tokenized_val_dataset = squad_dataset["validation"].map(preprocess_function, batched=True, remove_columns=squad_dataset["validation"].column_names)

    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Fine-tune the model
    training_args = TrainingArguments( 
        output_dir='./results',          # Output directory for model checkpoints 
        evaluation_strategy="epoch",     # Evaluate once per epoch 
        learning_rate=3e-5,              # Learning rate 
        per_device_train_batch_size=8,   # Batch size per device 
        per_device_eval_batch_size=16,   # Batch size for evaluation 
        num_train_epochs=3,              # Number of epochs 
        weight_decay=0.01,               # Weight decay to prevent overfitting 
    ) 

    trainer = Trainer( 
        model=model, 
        args=training_args, 
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
    ) 

 

    # Start training
    trainer.train()
    # Save the model
    saved_model_name = 'finetuned_squad_model'
    model.save_pretrained('./model_saved/%s' % saved_model_name)
    tokenizer.save_pretrained('./model_saved/%s' % saved_model_name)


if __name__ == "__main__":
    main()