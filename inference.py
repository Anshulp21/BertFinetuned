from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

# Check if a GPU is available, and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load your data and model as before
pd_data = pd.read_csv('./output_csv/output_1.csv')
pd_data = pd_data.astype(str)
pd_data = pd_data.dropna(subset=['transaction_description', 'taxonomy_node_label','code_category_description'], how='any')

unique_labels = pd_data['code_category_description'].unique()
label_dict = {i: label for i, label in enumerate(unique_labels)}

# Load the fine-tuned model and tokenizer
model_path = "./out/fine_tuned_model"  # Change to the path where you saved your fine-tuned model
model = BertForSequenceClassification.from_pretrained(model_path)
model = model.to(device)  # Move the model to the selected device
tokenizer = BertTokenizer.from_pretrained(model_path)

# Function to perform inference on a list of input texts
def perform_inference(input_texts, model, tokenizer, label_dict, threshold=0.3):
    results = {}
    for input_text in input_texts:
        # Tokenize the input text
        inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt', max_length=128)
        inputs = inputs.to(device)  # Move input tensors to the selected device

        # Perform inference on the appropriate device (GPU or CPU)
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predicted class probabilities
        predicted_probabilities = torch.softmax(outputs.logits, dim=1).tolist()[0]

        # Map class probabilities to label names based on the threshold
        predicted_labels = [label_dict[i] for i, prob in enumerate(predicted_probabilities) if prob > threshold]

        results[input_text] = predicted_labels

    return results

# List of input texts for inference
input_texts = ["Urine Delay-Complete Urinalysis Antech"]

# Perform inference on the list of input texts
inference_results = perform_inference(input_texts, model, tokenizer, label_dict, threshold=0.3)

# Print the results as a dictionary
print(inference_results)
