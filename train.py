from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import os

# Check if a GPU is available, and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pd_data = pd.read_csv('./output_csv/output_1.csv')
pd_data = pd_data.astype(str)

# Selecting First 10 records
old_keys = pd_data['transaction_description'].tolist()
new_keys = pd_data['taxonomy_node_label'].tolist()

# Step 1: Define and preprocess your dataset
data = {
    'legacy': old_keys,
    'new': new_keys
}

df = pd.DataFrame(data)

# Make unique
df = df.drop_duplicates()

df = df.dropna(subset=['legacy', 'new'], how='any')

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

model_name = "./out/fine_tuned_model"
# Step 2: Load the BERT tokenizer and model
if not os.path.exists("./out/fine_tuned_model"):
    model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(train_df['new'].unique()), ignore_mismatched_sizes=True)

# Move the model to the selected device (CPU or GPU)
model.to(device)

# Step 3: Tokenize the dataset and prepare it for fine-tuning
inputs = tokenizer(train_df['legacy'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)

# Convert labels to numerical values
label_dict = {label: idx for idx, label in enumerate(train_df['new'].unique())}
labels = torch.tensor([label_dict[label] for label in train_df['new'].tolist()], dtype=torch.long)

dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

# Step 4: Define training parameters
batch_size = 4
learning_rate = 2e-5
num_epochs = 3

optimizer = AdamW(model.parameters(), lr=learning_rate)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 5: Fine-tune the model
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids, attention_mask, labels = [item.to(device) for item in batch]  # Move tensors to the selected device
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# Step 6: Save the fine-tuned model
model.save_pretrained("./out/fine_tuned_model")
tokenizer.save_pretrained("./out/fine_tuned_model")
