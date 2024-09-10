get_ipython().system('pip install transformers')



#imports
import sys,os,os.path
import math
import random
import matplotlib.pyplot as plt
import time
import glob 
import copy
import enum
import numpy as np
from PIL import Image
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torchvision      
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import pandas as pd



#The below code is used as was provided in the Homework 9 lab manual handout.
import csv

sentences = []
sentiments = []
count = 0
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    # ignore the first line
    next(reader)
    for row in reader:
        sentences.append(row[0])
        sentiments.append(row[1])


print(sentences)


word_tokenized_sentences = [ sentence . split () for sentence in sentences ]
print ( word_tokenized_sentences [:2])

max_len = max ([len ( sentence ) for sentence in
word_tokenized_sentences ])
padded_sentences = [ sentence + ['[PAD]'] * ( max_len - len (
sentence ) ) for sentence in
word_tokenized_sentences ]
print ( padded_sentences [:2])

from transformers import DistilBertTokenizer

model_ckpt = "distilbert-base-uncased"
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
bert_tokenized_sentences_ids = [ distilbert_tokenizer.encode ( sentence , padding ='max_length',truncation =True ,max_length = max_len ) for sentence in sentences ]
print ( bert_tokenized_sentences_ids [:2])

bert_tokenized_sentences_tokens = [ distilbert_tokenizer.convert_ids_to_tokens (sentence ) for sentence in bert_tokenized_sentences_ids]
print ( bert_tokenized_sentences_tokens [:2])

vocab = {}
vocab['[PAD]'] = 0

print(vocab)

for sentence in padded_sentences:
    for token in sentence:
        if token not in vocab:
            vocab[token] = len(vocab)

# Convert the tokens to IDs
padded_sentences_ids = [[vocab[token] for token in sentence] for sentence in padded_sentences]
print(padded_sentences_ids[:2])

from transformers import DistilBertModel
import torch

model_name = 'distilbert-base-uncased'
distilbert_model = DistilBertModel.from_pretrained(model_name)
# Extract word embeddings
word_embeddings = []
# Convert padded sentence tokens into ids
for tokens in padded_sentences_ids:
    input_ids = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        outputs = distilbert_model(input_ids)
    word_embeddings.append(outputs.last_hidden_state)

print(word_embeddings[0].shape)

from transformers import DistilBertModel
import torch

model_name = 'distilbert-base-uncased'
distilbert_model = DistilBertModel.from_pretrained(model_name)
# Subword embeddings extraction
subword_embeddings = []
for tokens in bert_tokenized_sentences_ids:
    input_ids = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        outputs = distilbert_model(input_ids)
    subword_embeddings.append(outputs.last_hidden_state)

print(subword_embeddings[0].shape)



# Encode sentiments as one-hot vectors
def encode_sentiment(sentiment):
    if sentiment == 'positive':
        return np.array([1, 0, 0])
    elif sentiment == 'neutral':
        return np.array([0, 1, 0])
    elif sentiment == 'negative':
        return np.array([0, 0, 1])
    else:
        raise ValueError("Invalid sentiment value")


sentiments_one_hot =torch.tensor( [encode_sentiment(sentiment) for sentiment in sentiments])



# Define custom Sentiment Analysis Dataset class
class CustomSentimentAnalysisDataset(Dataset):
    def __init__(self, word_embeddings, sentiments):
        self.word_embeddings = word_embeddings
        self.sentiments = sentiments

    def __len__(self):
        return len(self.word_embeddings)

    def __getitem__(self, idx):
        word_embedding =(self.word_embeddings[idx])
        sentiment =(self.sentiments[idx])

        return word_embedding, sentiment



# Split the data into training and testing sets for word embeddings
from sklearn.model_selection import train_test_split
train_word_embeddings, test_word_embeddings, train_sentiments, test_sentiments = train_test_split(word_embeddings, sentiments_one_hot, test_size=0.2, random_state=42)

# Create DataLoader for training set for word embeddings
train_word_dataset = CustomSentimentAnalysisDataset(train_word_embeddings, train_sentiments)
train_word_dataloader = DataLoader(train_word_dataset, batch_size=1, shuffle=True)

# Create DataLoader for testing set for word embeddings
test_word_dataset = CustomSentimentAnalysisDataset(test_word_embeddings, test_sentiments)
test_word_dataloader = DataLoader(test_word_dataset, batch_size=1, shuffle=True)



# Split the data into training and testing sets for subword embeddings
from sklearn.model_selection import train_test_split
train_subword_embeddings, test_subword_embeddings, train_sentiments, test_sentiments = train_test_split(subword_embeddings, sentiments_one_hot, test_size=0.2, random_state=42)

# Create DataLoader for training set for subword embeddings
train_subword_dataset = CustomSentimentAnalysisDataset(train_subword_embeddings, train_sentiments)
train_subword_dataloader = DataLoader(train_subword_dataset, batch_size=1, shuffle=True)

# Create DataLoader for testing set for subword embeddings
test_subword_dataset = CustomSentimentAnalysisDataset(test_subword_embeddings, test_sentiments)
test_subword_dataloader = DataLoader(test_subword_dataset, batch_size=1, shuffle=True)




# Code for GRU network which is inspired from Dr. Avinash Kak's DL Studio Module with slight modifications
class GRUnet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, drop_prob=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        out = self.logsoftmax(out)
        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data
        #                                     batch_size   
        hidden = weight.new(  self.num_layers,     1,         self.hidden_size   ).zero_()
        return hidden




# Code for GRU training routine which is inspired from Dr. Avinash Kak's DL Studio Module with slight modifications
def run_code_for_training_with_GRU( net,train_dataloader,device,model_name, display_train_loss=True):        
    filename_for_out = "performance_numbers_GRU_" + str(10) + ".txt"
    FILE = open(filename_for_out, 'w')
    net.to(device)

    criterion = nn.NLLLoss()
    accum_times = []
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas = (0.8, 0.999))
    start_time = time.perf_counter()
    training_loss_tally = []
    for epoch in range(10):  
        print("")
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):    
            hidden = net.init_hidden().to(device)     
            
            review_tensor,sentiment = data
            review_tensor = review_tensor[0].to(device)
            sentiment = sentiment[0].to(device)

            optimizer.zero_grad()
            for k in range(review_tensor.shape[1]):

                output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0).to(device), hidden)
            loss = criterion(output, torch.argmax(sentiment.unsqueeze(0),1))
            running_loss += loss.item()
            loss.backward(retain_graph=True)        
            optimizer.step()

            if i % 200 == 199:    
                avg_loss = running_loss / float(200)
                training_loss_tally.append(avg_loss)
                current_time = time.perf_counter()
                time_elapsed = current_time-start_time
                print("[epoch:%d  iter:%4d  elapsed_time: %4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                accum_times.append(current_time-start_time)
                FILE.write("%.3f\n" % avg_loss)
                FILE.flush()
                running_loss = 0.0
    print("Total Training Time: {}".format(str(sum(accum_times))))
    print("\nFinished Training\n")
    torch.save(net.state_dict(), f"{model_name}.pt")
    if display_train_loss:
        plt.figure(figsize=(10,5))
        plt.title("Training Loss vs. Iterations for unidirectional GRU using word embeddings")
        plt.plot(training_loss_tally)
        plt.xlabel("iterations")
        plt.ylabel("training loss")

        plt.legend(["Plot of loss versus iterations"], fontsize="x-large")
        plt.savefig("training_loss.png")
        plt.show()
    return training_loss_tally




# Code for GRU testing routine which is inspired from Dr. Avinash Kak's DL Studio Module with slight modifications
def run_code_for_testing_text_classification_with_GRU(net, test_dataloader,model_name, device):
    net.load_state_dict(torch.load(f"{model_name}.pt"))
    net.to(device)
    classification_accuracy = 0.0
    negative_total = 0
    positive_total = 0
    neutral_total = 0
    confusion_matrix = torch.zeros(3, 3)
    
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            review_tensor, sentiment = data
            review_tensor = review_tensor[0].to(device)
            sentiment = sentiment[0].to(device)
            hidden = net.init_hidden().to(device)
            
            for k in range(review_tensor.shape[1]):
                output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                
            predicted_idx = torch.argmax(output).item()
            gt_idx = torch.argmax(sentiment).item()
            if i % 100 == 99:
                print("   [i=%d]    predicted_label=%d       gt_label=%d\n\n" % (i+1, predicted_idx, gt_idx))
            
            if predicted_idx == gt_idx:
                classification_accuracy += 1
            
            if gt_idx == 0: 
                positive_total += 1
            elif gt_idx == 1:
                neutral_total += 1
            elif gt_idx == 2:
                negative_total += 1
                
            confusion_matrix[gt_idx, predicted_idx] += 1
            
    classification_accuracy /= len(test_dataloader)
    print("\nOverall classification accuracy: %0.2f%%" % (classification_accuracy * 100))
    
    out_percent = np.zeros((3,3), dtype='float')
    out_percent[0,:] = 100 * confusion_matrix[0,:] / negative_total
    out_percent[1,:] = 100 * confusion_matrix[1,:] / neutral_total
    out_percent[2,:] = 100 * confusion_matrix[2,:] / positive_total
    
    print("\n\nNumber of negative reviews tested: %d" % negative_total)
    print("\nNumber of neutral reviews tested: %d" % neutral_total)
    print("\nNumber of positive reviews tested: %d" % positive_total)
    
    print("\n\nDisplaying the confusion matrix:\n")


    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['positive', 'neutral', 'negative'],
                yticklabels=['positive', 'neutral', 'negative'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for word embeddings for Unidirectional GRU')#edit the name of corresponding model for title label
    plt.show()


# Initialize the Unidirectional GRU model
model = GRUnet(768, hidden_size=100, output_size=3, num_layers=3)
device=torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)




## TRAINING for unidirectional GRU for subword embeddings:
avg_loss_uni_subword= run_code_for_training_with_GRU(net=model,train_dataloader=train_subword_dataloader, device=device, model_name ="unigru_subword", display_train_loss=True)

## Testing for unidirectional GRU for subword embeddings:
run_code_for_testing_text_classification_with_GRU(net=model,test_dataloader=test_subword_dataloader, model_name="unigru_subword", device=device)




## TRAINING for unidirectional GRU for word embeddings:
avg_loss_uni_word= run_code_for_training_with_GRU(net=model,train_dataloader=train_word_dataloader, device=device, model_name ="unigru_word", display_train_loss=True)

## TESTING for unidirectional GRU for word embeddings:
run_code_for_testing_text_classification_with_GRU(net=model,test_dataloader=test_word_dataloader, model_name="unigru_word", device=device)



# Code for Bidirectional GRU Network which is inspired from Dr. Avinash Kak's DL Studio Module with slight modifications
# to the Unidirectional GRU
class BiGRUnet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, drop_prob=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=drop_prob, bidirectional=True) # True for bidirectional GRU
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        out = self.fc(self.relu(out))
        out = self.logsoftmax(out)
        return out, h

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters()).data
        num_directions = 2 if self.bidirectional else 1
        hidden = weight.new(self.num_layers * num_directions, 1, self.hidden_size).zero_()
        return hidden




# Code for Bidirectional GRU training routine which is inspired from Dr. Avinash Kak's DL Studio Module with slight modifications
def run_code_for_training_with_BiGRU( net,train_dataloader,device,model_name, display_train_loss=True):        
    filename_for_out = "performance_numbers_BiGRU" + str(10) + ".txt"
    FILE = open(filename_for_out, 'w')
    net.to(device)
    criterion = nn.NLLLoss()
    accum_times = []
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas = (0.8, 0.999))
    start_time = time.perf_counter()
    training_loss_tally = []
    for epoch in range(10):  
        print("")
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):    
            hidden = net.init_hidden().to(device)     
            
            review_tensor,sentiment = data
            review_tensor = review_tensor[0].to(device)
            sentiment = sentiment[0].to(device)

            optimizer.zero_grad()
            for k in range(review_tensor.shape[1]):
                output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0).to(device), hidden)
            loss = criterion(output, torch.argmax(sentiment.unsqueeze(0),1))
            running_loss += loss.item()
            loss.backward(retain_graph=True)        
            optimizer.step()

            if i % 200 == 199:    
                avg_loss = running_loss / float(200)
                training_loss_tally.append(avg_loss)
                current_time = time.perf_counter()
                time_elapsed = current_time-start_time
                print("[epoch:%d  iter:%4d  elapsed_time: %4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                accum_times.append(current_time-start_time)
                FILE.write("%.3f\n" % avg_loss)
                FILE.flush()
                running_loss = 0.0
    print("Total Training Time: {}".format(str(sum(accum_times))))
    print("\nFinished Training\n")
    torch.save(net.state_dict(), f"{model_name}.pt")
    if display_train_loss:
        plt.figure(figsize=(10,5))
        plt.title("Training Loss vs. Iterations for Bidirectional GRU using word embeddings") #edit the name of corresponding model for title label
        plt.plot(training_loss_tally)
        plt.xlabel("iterations")
        plt.ylabel("training loss")
        plt.legend(["Plot of loss versus iterations"], fontsize="x-large")
        plt.savefig("training_loss.png")
        plt.show()
    return training_loss_tally




# Code for Bidirectional GRU testing routine which is inspired from Dr. Avinash Kak's DL Studio Module with slight modifications
def run_code_for_testing_text_classification_with_BiGRU(net, test_dataloader,model_name, device):
    net.load_state_dict(torch.load(f"{model_name}.pt"))
    net.to(device)
    classification_accuracy = 0.0
    negative_total = 0
    positive_total = 0
    neutral_total = 0
    confusion_matrix = torch.zeros(3, 3)
    
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            review_tensor, sentiment = data
            review_tensor = review_tensor[0].to(device)
            sentiment = sentiment[0].to(device)
            hidden = net.init_hidden().to(device)
            
            for k in range(review_tensor.shape[1]):
                output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                
            predicted_idx = torch.argmax(output).item()
            gt_idx = torch.argmax(sentiment).item()
            if i % 100 == 99:
                print("   [i=%d]    predicted_label=%d       gt_label=%d\n\n" % (i+1, predicted_idx, gt_idx))
            
            if predicted_idx == gt_idx:
                classification_accuracy += 1
            
            if gt_idx == 0: 
                positive_total += 1
            elif gt_idx == 1:
                neutral_total += 1
            elif gt_idx == 2:
                negative_total += 1
                
            confusion_matrix[gt_idx, predicted_idx] += 1
            
    classification_accuracy /= len(test_dataloader)
    print("\nOverall classification accuracy: %0.2f%%" % (classification_accuracy * 100))
    
    out_percent = np.zeros((3,3), dtype='float')
    out_percent[0,:] = 100 * confusion_matrix[0,:] / negative_total
    out_percent[1,:] = 100 * confusion_matrix[1,:] / neutral_total
    out_percent[2,:] = 100 * confusion_matrix[2,:] / positive_total
    
    print("\n\nNumber of negative reviews tested: %d" % negative_total)
    print("\nNumber of neutral reviews tested: %d" % neutral_total)
    print("\nNumber of positive reviews tested: %d" % positive_total)
    
    print("\n\nDisplaying the confusion matrix:\n")


    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['positive', 'neutral', 'negative'],
                yticklabels=['positive', 'neutral', 'negative'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for subword embeddings for Bidirectional GRU')#edit the name of corresponding model for title label
    plt.show()




# Initialize the Bidirectional BiGRU model
model1 = BiGRUnet(768, hidden_size=100, output_size=3, num_layers=3)
device=torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
number_of_learnable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)

num_layers = len(list(model1.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)




## TRAINING for bidirectional GRU for subword embeddings:
avg_loss_bidirec_subword= run_code_for_training_with_BiGRU(net=model1,train_dataloader=train_subword_dataloader, device=device, model_name ="bigru_subword", display_train_loss=True)

## TESTING for bidirectional GRU for subword embeddings:
run_code_for_testing_text_classification_with_BiGRU(net=model1,test_dataloader=test_subword_dataloader, model_name="bigru_subword", device=device)





## TRAINING for bidirectional GRU for word embeddings:
avg_loss_bidirec_word= run_code_for_training_with_BiGRU(net=model1,train_dataloader=train_word_dataloader, device=device, model_name ="bigru_word", display_train_loss=True)


## TESTING for bidirectional GRU for word embeddings:
run_code_for_testing_text_classification_with_BiGRU(net=model1,test_dataloader=test_word_dataloader, model_name="bigru_word", device=device)

