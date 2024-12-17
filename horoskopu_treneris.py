from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json

# Ielādē modeli un tokenizeri
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Iestatiet aizpildes simbolu
tokenizer.pad_token = tokenizer.eos_token

# Ielādē datus un sagatavo
with open("horoscope_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Apstrādā datus
processed_data = [
    {
        "text": f"Zodiaka zīme: {item['zime']}. Datums: {item['datums']}. Horoskops: {item['horoskops']}."
    }
    for item in raw_data  # raw_data satur jūsu oriģinālos datus
]

# Pārvērst datus par Hugging Face Dataset
dataset = Dataset.from_list(processed_data)

# Tokenizācijas funkcija
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

# Tokenizēt un noņemt nevajadzīgos laukus
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])  # Noņem 'text' kolonnu

# Treniņa parametri
training_args = TrainingArguments(
    output_dir="./model_output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    remove_unused_columns=False
)

# Treneris
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer
)

# Apmācības sākums
trainer.train()

# Saglabā apmācīto modeli un tokenizeri
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")