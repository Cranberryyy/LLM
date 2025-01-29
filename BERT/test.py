import sys
import torch
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification
from transformers import BertForQuestionAnswering
sys.path.append('./model_saved')


model_name = "bert-base-uncased"
model_name = "finetuned_squad_model"
def decode_output(outputs):
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_indexes = torch.argmax(start_logits)
    end_indexes = torch.argmax(end_logits)
    answer_tokens = inputs["input_ids"][0][start_indexes:end_indexes+1]
    answer = tokenizer.decode(answer_tokens)
    return answer

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_text = "Who was Jim Henson ?"
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer = decode_output(outputs)
print("predicted answer: %s " %answer)
