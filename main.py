from transformers import AutoTokenizer, MT5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import ipdb
import json

tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")
print(model)

filename = "data.jsonl"
with open(filename, 'r', encoding='utf-8') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
print(data)


prefix = "Complete the output: "
max_input_length = 128
def preprocess_function(examples):
    # ipdb.set_trace()
    inputs = [prefix + examples["instruction"] +" "+ examples["input"] + " Output: "]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding = 'max_length', return_tensors="pt")
    model_inputs["input_ids"] = model_inputs["input_ids"].squeeze(0)
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer([examples["output"]], max_length=max_input_length, truncation=True, padding = 'max_length', return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = model_inputs["labels"].squeeze(0)
    return model_inputs


tokenized_datasets = [preprocess_function(d) for d in data]

model = model.to("cuda")
print(len(tokenized_datasets))
# ipdb.set_trace()
trainer = Seq2SeqTrainer(
            model=model,
            train_dataset=tokenized_datasets,
            eval_dataset=tokenized_datasets,
            tokenizer=tokenizer,
            args=Seq2SeqTrainingArguments(
                output_dir='results_large',
                logging_strategy="steps",
                logging_steps=5,
                predict_with_generate=True,
                overwrite_output_dir=True,
                do_train=True,
                do_eval=True,
                eval_steps=50,
                num_train_epochs=10,
                save_steps = 200,
                generation_max_length=max_input_length,
                per_device_train_batch_size=8,
            )
        )
trainer.train()

model.save_pretrained("results_large")
tokenizer.save_pretrained("results_large")

# Conditional generation

