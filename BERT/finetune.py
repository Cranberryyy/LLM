import sys
import torch
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification
from transformers import BertForQuestionAnswering
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
sys.path.append('./model_saved')


def preprocess_data(examples): 
    import pdb; pdb.set_trace()
    # Tokenize the context and question 
    inputs = tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512) 
    # Find the start and end positions of the answer 
    start_positions = [] 
    end_positions = [] 
    for i in range(len(examples['answers']['text'])): 
        answer = examples['answers']['text'][i] 
        start_idx = examples['context'].find(answer) 
        end_idx = start_idx + len(answer) 
        start_positions.append(start_idx) 
        end_positions.append(end_idx) 

    # Add the labels for start and end positions 
    inputs['start_positions'] = start_positions 
    inputs['end_positions'] = end_positions 
    return inputs 

 



model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = load_dataset('squad')
# Apply preprocessing to the dataset 
train_dataset = dataset['train'].map(preprocess_data, batched=True) 
val_dataset = dataset['validation'].map(preprocess_data, batched=True) 

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
    train_dataset=train_dataset, 
    eval_dataset=val_dataset, 
) 

 

# Start training
trainer.train()
# Save the model
saved_model_name = 'finetuned_squad_model'
model.save_pretrained('./model_saved/%s' % saved_model_name)
tokenizer.save_pretrained('./model_saved/%s' % saved_model_name)
