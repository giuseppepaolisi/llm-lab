from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
small_train = dataset["train"].select(range(500))
small_eval = dataset["validation"].select(range(100))
print(f"Dataset pronto: {len(small_train)}")

# Caricamento GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # tutti i modelli hanno un carattere per il padding specifico (serve a riempire le sequenze più corte per farle avere tutte la stessa lunghezza)
# gpt-2 essendo un modello generativo non ha un token di padding predefinito, gli assegniamo il suo eos token (end of string)
model = GPT2LMHeadModel.from_pretrained("gpt2") # Andiamo a scaricare il modello

# Assicuriamoci che i dati siano nel formato corretto per il training
def prepare_data(exemples):
    # convertiamo il testo
    inputs = tokenizer(
        exemples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

train_data = small_train.map(prepare_data, batched=True)
eval_data = small_eval.map(prepare_data, batched=True)

# Configuriamo il training
print("\nConfigurazione training...")
training_args = TrainingArguments(
    output_dir="./risultati",          # Dove salvare il modello
    num_train_epochs=3,                # Numero di epoch, indica quante volte il modello vedrà l'intero dataset di training (un epoch completa indica che tutti gli esempi sono stati utilizzati una volta per aggiornare i pesi del modello)
    per_device_train_batch_size=16,     # Batch size, nummero di esempi che il modello elabora contemporaneamente prima di aggiornare i pesi
    per_device_eval_batch_size=16,      # Batch più grande velocizza il processo ma richiede più memoria della gpu
    warmup_steps=100,                  # Warmup steps, legato al learning rate (quanto velocemente il modello impara)
    logging_steps=50,                  # Log ogni 50 step
    save_steps=500,                    # Salva ogni 500 step
    eval_strategy="steps",             # Valuta durante il training (CAMBIATO DA evaluation_strategy)
    eval_steps=100,                    # Valuta ogni 100 step
    load_best_model_at_end=True,       # Carica il miglior modello alla fine
    metric_for_best_model="eval_loss", # Metrica per determinare il miglior modello
)

# Creiamo il trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
)

# Avviamo il training
print("🚀 Inizio training...")
trainer.train()
print("\n✅ Training completato!")

# Salviamo il modello
model.save_pretrained("./models/gpt2-mio")
tokenizer.save_pretrained("./models/gpt2-mio")
print("✅ Modello salvato in ./models/gpt2-mio")

# Testiamo il modello
print("\n🧪 Test del modello fine-tuned...")
from transformers import pipeline

# Creiamo un generatore di testo
generator = pipeline("text-generation", model="./models/gpt2-mio", tokenizer=tokenizer)

# Test con alcuni prompt
test_prompts = [
    "The future of artificial intelligence",
    "In a world where technology",
    "The most important discovery"
]

for prompt in test_prompts:
    print(f"\n📝 Prompt: '{prompt}'")
    result = generator(prompt, max_length=50, num_return_sequences=1, temperature=0.7)
    print(f"💬 Generazione: {result[0]['generated_text']}")
    print("-" * 50)