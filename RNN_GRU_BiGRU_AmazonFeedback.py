
#imports
import os
import torch
import random
import numpy as np
import requests
import matplotlib.pyplot as plt
import sys
import torch.nn as nn
import torch.optim as optim
import copy
import time
from tqdm import tqdm
import gzip
import pickle
import gensim.downloader as gen_api
import gensim.downloader as genapi
from gensim.models import KeyedVectors
import seaborn as sns


seed = 10
random.seed(seed)
np.random.seed(seed)

# Code for dataset class which is inspired from Dr. Avinash Kak's DL Studio Module with slight modifications.
class SentimentAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_file, mode = 'train', path_to_saved_embeddings=None):
        super(SentimentAnalysisDataset, self).__init__()
        self.path_to_saved_embeddings = path_to_saved_embeddings
        self.mode = mode
        root_dir = root
        f = gzip.open(root_dir + dataset_file, 'rb')
        dataset = f.read()
        if path_to_saved_embeddings is not None:
            if os.path.exists(path_to_saved_embeddings + 'vectors.kv'):
                self.word_vectors = KeyedVectors.load(path_to_saved_embeddings + 'vectors.kv')
            else:
                self.word_vectors = genapi.load("word2vec-google-news-300")

                self.word_vectors.save(path_to_saved_embeddings + 'vectors.kv')
        if mode == 'train':
            if sys.version_info[0] == 3:
                self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
            else:
                self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
            self.categories = sorted(list(self.positive_reviews_train.keys()))
            self.category_sizes_train_pos = {category : len(self.positive_reviews_train[category]) for category in self.categories}
            self.category_sizes_train_neg = {category : len(self.negative_reviews_train[category]) for category in self.categories}
            self.indexed_dataset_train = []
            for category in self.positive_reviews_train:
                for review in self.positive_reviews_train[category]:
                    self.indexed_dataset_train.append([review, category, 1])
            for category in self.negative_reviews_train:
                for review in self.negative_reviews_train[category]:
                    self.indexed_dataset_train.append([review, category, 0])
            random.shuffle(self.indexed_dataset_train)
        elif mode == 'test':
            if sys.version_info[0] == 3:
                self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
            else:
                self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
            self.vocab = sorted(self.vocab)
            self.categories = sorted(list(self.positive_reviews_test.keys()))
            self.category_sizes_test_pos = {category : len(self.positive_reviews_test[category]) for category in self.categories}
            self.category_sizes_test_neg = {category : len(self.negative_reviews_test[category]) for category in self.categories}
            self.indexed_dataset_test = []
            for category in self.positive_reviews_test:
                for review in self.positive_reviews_test[category]:
                    self.indexed_dataset_test.append([review, category, 1])
            for category in self.negative_reviews_test:
                for review in self.negative_reviews_test[category]:
                    self.indexed_dataset_test.append([review, category, 0])
            random.shuffle(self.indexed_dataset_test)

    def review_to_tensor(self, review):
        list_of_embeddings = []
        for i,word in enumerate(review):
            if word in self.word_vectors.key_to_index:
                embedding = self.word_vectors[word]
                list_of_embeddings.append(np.array(embedding))
            else:
                next
        review_tensor = torch.FloatTensor(list_of_embeddings)
        return review_tensor

    def sentiment_to_tensor(self, sentiment):
        sentiment_tensor = torch.zeros(2)
        if sentiment == 1:
            sentiment_tensor[1] = 1
        elif sentiment == 0:
            sentiment_tensor[0] = 1
        sentiment_tensor = sentiment_tensor.type(torch.long)
        return sentiment_tensor

    def __len__(self):
        if self.mode == 'train':
            return len(self.indexed_dataset_train)
        elif self.mode == 'test':
            return len(self.indexed_dataset_test)

    def __getitem__(self, idx):
        sample = self.indexed_dataset_train[idx] if self.mode == 'train' else self.indexed_dataset_test[idx]
        review = sample[0]
        review_category = sample[1]
        review_sentiment = sample[2]
        review_sentiment = self.sentiment_to_tensor(review_sentiment)
        review_tensor = self.review_to_tensor(review)
        category_index = self.categories.index(review_category)
        sample = {'review'       : review_tensor,
                    'category'     : category_index, # should be converted to tensor, but not yet used
                    'sentiment'    : review_sentiment }
        return sample

# Code for torch.nn GRU with Embeddings inspired from Dr. Avinash Kak's DL Studio Module.
class GRUnetWithEmbeddings(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUnetWithEmbeddings, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[-1]))
        out = self.logsoftmax(out)
        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data
        #                  num_layers  batch_size    hidden_size
        hidden = weight.new(  2,          1,         self.hidden_size    ).zero_()
        return hidden

# Code for Bidirectional GRU Network with Embeddings which is inspired from Dr. Avinash Kak's DL Studio Module
# with slight modifications to the Unidirectional GRU
class BiGRUnetWithEmbeddings(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BiGRUnetWithEmbeddings, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[-1]))
        out = self.logsoftmax(out)
        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data
        #                  num_layers  batch_size    hidden_size
        hidden = weight.new(  2*2,          1,         self.hidden_size    ).zero_()
        return hidden

#Code for GRU with embeddings training routine which is inspired from Dr. Avinash Kak's DL Studio Module with slight modifications.
def run_code_for_training_for_text_classification_with_GRU(device, net, dataloader, model_name, epochs, display_interval):
    net = net.to(device)
    criterion = nn.NLLLoss()
    accum_times = []
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas = (0.5, 0.999))
    training_loss_tally = []
    start_time = time.perf_counter()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
            review_tensor = review_tensor.to(device)
            sentiment = sentiment.to(device)

            optimizer.zero_grad()
            hidden = net.init_hidden().to(device)
            output, hidden = net(torch.unsqueeze(review_tensor[0], 1), hidden)
            loss = criterion(output, torch.argmax(sentiment, 1))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i+1) % display_interval == 0:
                avg_loss = running_loss / float(display_interval)
                training_loss_tally.append(avg_loss)
                current_time = time.perf_counter()
                time_elapsed = current_time-start_time
                print("[epoch:%d  iter:%4d  elapsed_time:%4d secs] loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                accum_times.append(current_time-start_time)
                running_loss = 0.0

    torch.save(net.state_dict(), os.path.join('/content/drive/My Drive/saved_models',f'{model_name}.pt'))

    print("Total Training Time: {}".format(str(sum(accum_times))))
    print("\nFinished Training\n\n")
    plt.figure(figsize=(10,5))
    plt.title(f"Training Loss vs. Iterations - {model_name}")
    plt.plot(training_loss_tally)
    plt.xlabel("Iterations")
    plt.ylabel("Training loss")
    plt.legend()
    plt.savefig(f"/content/drive/My Drive/training_loss_{model_name}.png")
    plt.show()

    return training_loss_tally

#Code for GRU with embeddings testing routine which is inspired from Dr. Avinash Kak's DL Studio Module with slight modifications.
def run_code_for_testing_text_classification_with_GRU(device, net, model_path, dataloader, model_name, display_interval):
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    classification_accuracy = 0.0
    negative_total = 0
    positive_total = 0
    confusion_matrix = torch.zeros(2,2)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
            review_tensor = review_tensor.to(device)
            sentiment = sentiment.to(device)

            hidden = net.init_hidden().to(device)
            output, hidden = net(torch.unsqueeze(review_tensor[0], 1), hidden)
            predicted_idx = torch.argmax(output).item()
            gt_idx = torch.argmax(sentiment).item()
            if (i+1) % display_interval == 0:
                print("   [i=%d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
            if predicted_idx == gt_idx:
                classification_accuracy += 1
            if gt_idx == 0:
                negative_total += 1
            elif gt_idx == 1:
                positive_total += 1
            confusion_matrix[gt_idx,predicted_idx] += 1
    print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
    out_percent = np.zeros((2,2), dtype='float')
    out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
    out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
    out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
    out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
    print("\n\nNumber of positive reviews tested: %d" % positive_total)
    print("\n\nNumber of negative reviews tested: %d" % negative_total)
    print("\n\nDisplaying the confusion matrix:\n")
    out_str = "                      "
    out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
    print(out_str + "\n")
    for i,label in enumerate(['true negative', 'true positive']):
        out_str = "%12s:  " % label
        for j in range(2):
            out_str +=  "%18s%%" % out_percent[i,j]
        print(out_str)

    labels = []
    classes=['negative_reviews', 'positive_reviews']
    num_classes = len(classes)
    for row in range(num_classes):
        rows = []
        total_labels =  np.sum(confusion_matrix.numpy()[row])
        for col in range(num_classes):
            count = confusion_matrix.numpy()[row][col]
            label = str(count)
            rows.append(label)
        labels.append(rows)
    labels = np.asarray(labels)

    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix.numpy(), annot=labels, fmt="", cmap="Blues", cbar=True,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for model {model_name}')
    plt.show()

# training dataset for sentiment_dataset_train_200
train_dataset_200 = SentimentAnalysisDataset('/content/drive/My Drive/','sentiment_dataset_train_200.tar.gz',
                                         path_to_saved_embeddings = '/content/drive/My Drive/word2vec/')



# testing dataset for sentiment_dataset_train_200
test_dataset_200 = SentimentAnalysisDataset('/content/drive/My Drive/', 'sentiment_dataset_test_200.tar.gz',
                                       mode = 'test', path_to_saved_embeddings = '/content/drive/My Drive/word2vec/')



# Create custom training/testing dataloader for sentiment_dataset_train_200
train_data_loader = torch.utils.data.DataLoader(train_dataset_200, batch_size=1, shuffle=True, num_workers=1)
test_data_loader = torch.utils.data.DataLoader(test_dataset_200, batch_size=1, shuffle=True, num_workers=1)

# Initialize GRU with Embeddings
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")


model = GRUnetWithEmbeddings(input_size=300, hidden_size=100, output_size=2, num_layers=2)

epochs = 4
display_interval = 500

# Count the number of learnable parameters and layers
number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nThe number of learnable parameters in the model:", number_of_learnable_params)


num_layers = len(list(model.parameters()))
print("\nThe number of layers in the model:", num_layers)

# Training for sentiment_dataset_train_200
net1_losses = run_code_for_training_for_text_classification_with_GRU(device, model, dataloader=train_data_loader,
                                        model_name='HW9_extra_200', epochs=epochs, display_interval=display_interval)

# Testing for sentiment_dataset_train_200
save_path = '/content/drive/My Drive/saved_models/HW9_extra_200.pt'
run_code_for_testing_text_classification_with_GRU(device, model, dataloader=test_data_loader,
                             display_interval=display_interval, model_path=save_path, model_name='HW9_extra_200.pt')

# Initialize BiGRU with Embeddings
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")

model = BiGRUnetWithEmbeddings(input_size=300, hidden_size=100, output_size=2, num_layers=2)

epochs = 4
display_interval = 500

# Count the number of learnable parameters
number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nThe number of learnable parameters in the model:", number_of_learnable_params)

# Count the number of layers
num_layers = len(list(model.parameters()))
print("\nThe number of layers in the model:", num_layers)

# Training for sentiment_dataset_train_200
net1_losses = run_code_for_training_for_text_classification_with_GRU(device, model, dataloader=train_data_loader,
                                        model_name='HW9_extra_bi200', epochs=epochs, display_interval=display_interval)

# Testing for sentiment_dataset_train_200
save_path = '/content/drive/My Drive/saved_models/HW9_extra_bi200.pt'
run_code_for_testing_text_classification_with_GRU(device, model, dataloader=test_data_loader,
                             display_interval=display_interval, model_path=save_path, model_name='HW9_extra_bi200.pt')

# training dataset for sentiment_dataset_train_400
train_dataset_400 = SentimentAnalysisDataset('/content/drive/My Drive/','sentiment_dataset_train_400.tar.gz',
                                         path_to_saved_embeddings = '/content/drive/My Drive/word2vec/')



# testing dataset for sentiment_dataset_train_400
test_dataset_400 = SentimentAnalysisDataset('/content/drive/My Drive/', 'sentiment_dataset_test_400.tar.gz',
                                       mode = 'test', path_to_saved_embeddings = '/content/drive/My Drive/word2vec/')



# Create custom training/validation dataloader
train_data_loader = torch.utils.data.DataLoader(train_dataset_400, batch_size=1, shuffle=True, num_workers=1)
test_data_loader = torch.utils.data.DataLoader(test_dataset_400, batch_size=1, shuffle=True, num_workers=1)

# Initialize GRU with Embeddings
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")

model = GRUnetWithEmbeddings(input_size=300, hidden_size=100, output_size=2, num_layers=2)


epochs = 2
display_interval = 500

# Count the number of learnable parameters
number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nThe number of learnable parameters in the model:", number_of_learnable_params)

# Count the number of layers
num_layers = len(list(model.parameters()))
print("\nThe number of layers in the model:", num_layers)

# training for sentiment_dataset_train_400
net1_losses = run_code_for_training_for_text_classification_with_GRU(device, model, dataloader=train_data_loader,
                                        model_name='HW9_extra_400', epochs=epochs, display_interval=display_interval)

# testing for sentiment_dataset_train_400
save_path = '/content/drive/My Drive/saved_models/HW9_extra_400.pt'
run_code_for_testing_text_classification_with_GRU(device, model, dataloader=test_data_loader,
                             display_interval=display_interval, model_path=save_path, model_name='HW9_extra_400.pt')

# Initialize BiGRU with Embeddings
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")

model = BiGRUnetWithEmbeddings(input_size=300, hidden_size=100, output_size=2, num_layers=2)


epochs = 2
display_interval = 500

# Count the number of learnable parameters
number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nThe number of learnable parameters in the model:", number_of_learnable_params)

# Count the number of layers
num_layers = len(list(model.parameters()))
print("\nThe number of layers in the model:", num_layers)

# training for sentiment_dataset_train_400
net1_losses = run_code_for_training_for_text_classification_with_GRU(device, model, dataloader=train_data_loader,
                                        model_name='HW9_extra_bi400', epochs=epochs, display_interval=display_interval)

# training for sentiment_dataset_train_400
save_path = '/content/drive/My Drive/saved_models/HW9_extra_bi400.pt'
run_code_for_testing_text_classification_with_GRU(device, model, dataloader=test_data_loader,
                             display_interval=display_interval, model_path=save_path, model_name='HW9_extra_bi400.pt')

