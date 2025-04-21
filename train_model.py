import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.5):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.dropout(output)
        return output

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.5):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.dropout(output)
        return output

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, 
                          batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.dropout(output)
        return output

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.2):
        super(BiRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.dropout(output)
        
        return output

class CrossAttention(nn.Module):
    def __init__(self, input_size, dropout_rate=0.2, output_size=128):
        super(CrossAttention, self).__init__()
        self.query_proj_1 = nn.Linear(input_size, input_size)
        self.key_proj_1 = nn.Linear(input_size, input_size)
        self.value_proj_1 = nn.Linear(input_size, input_size)
        self.query_proj_2 = nn.Linear(input_size, input_size)
        self.key_proj_2 = nn.Linear(input_size, input_size)
        self.value_proj_2 = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(input_size, output_size)

    def forward(self, seq1, seq2):
        # Project input sequences to Q, K, V
        query1 = self.query_proj_1(seq1)
        key2 = self.key_proj_1(seq2)
        value2 = self.value_proj_1(seq2)

        # Compute attention scores for seq1 attending to seq2
        attention_scores_1 = torch.matmul(query1, key2.transpose(-2, -1)) / (seq1.size(-1) ** 0.5)
        attention_weights_1 = self.softmax(attention_scores_1)
        attention_weights_1 = self.dropout(attention_weights_1)  # Apply dropout to the attention weights
        output1_attended = torch.matmul(attention_weights_1, value2)

        # Project input sequences to Q, K, V for the reverse direction
        query2 = self.query_proj_2(seq2)
        key1 = self.key_proj_2(seq1)
        value1 = self.value_proj_2(seq1)

        # Compute attention scores for seq2 attending to seq1
        attention_scores_2 = torch.matmul(query2, key1.transpose(-2, -1)) / (seq2.size(-1) ** 0.5)
        attention_weights_2 = self.softmax(attention_scores_2)
        attention_weights_2 = self.dropout(attention_weights_2)  # Apply dropout to the attention weights
        output2_attended = torch.matmul(attention_weights_2, value1)

        # Combine the attended outputs and project them to the desired output size
        combined_output1 = self.output_proj(output1_attended)
        combined_output2 = self.output_proj(output2_attended)

        return combined_output1, combined_output2

class LSTMMerge(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5):
        super(LSTMMerge, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, X_ag, X_ab):
        output1, _ = self.lstm1(X_ag)
        output2, _ = self.lstm2(X_ab)

        output1 = self.dropout(output1)
        output2 = self.dropout(output2)

        result = torch.matmul(output1, output2.transpose(1, 2))
        
        return result

class BiLSTMMerge(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5):
        super(BiLSTMMerge, self).__init__()
        self.bilstm1 = nn.LSTM(input_size, hidden_size, num_layers,bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(input_size, hidden_size, num_layers,bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, X_ag, X_ab):
        output1, _ = self.bilstm1(X_ag)
        output2, _ = self.bilstm2(X_ab)

        output1 = self.dropout(output1)
        output2 = self.dropout(output2)

        result = torch.matmul(output1, output2.transpose(1, 2))
        
        return result

class RNNMerge(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5):
        super(RNNMerge, self).__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn2 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, X_ag, X_ab):
        output1, _ = self.rnn1(X_ag)
        output2, _ = self.rnn2(X_ab)

        output1 = self.dropout(output1)
        output2 = self.dropout(output2)

        result = torch.matmul(output1, output2.transpose(1, 2))
        
        return result

class BiRNNMerge(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5):
        super(BiRNNMerge, self).__init__()
        self.birnn1 = nn.RNN(input_size, hidden_size, num_layers,bidirectional=True, batch_first=True)
        self.birnn2 = nn.RNN(input_size, hidden_size, num_layers,bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, X_ag, X_ab):
        output1, _ = self.birnn1(X_ag)
        output2, _ = self.birnn2(X_ab)

        output1 = self.dropout(output1)
        output2 = self.dropout(output2)

        result = torch.matmul(output1, output2.transpose(1, 2))
        
        return result

class CAT_BiRNN(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, output_size=128, num_layers=1, dropout_rate=0.5):
        super(CAT_BiRNN, self).__init__()
        self.cat = CrossAttention(input_size, dropout_rate, output_size)
        self.birnn1 = nn.RNN(output_size, hidden_size, num_layers,bidirectional=True, batch_first=True)
        self.birnn2 = nn.RNN(output_size, hidden_size, num_layers,bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X_ag, X_ab):
        ag_output, ab_output = self.cat(X_ag, X_ab)
        output1, _ = self.birnn1(ag_output)
        output2, _ = self.birnn2(ab_output)

        output1 = self.dropout(output1)
        output2 = self.dropout(output2)

        result = torch.matmul(output1, output2.transpose(1, 2))

        return result

class CAT_BiLSTM(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, output_size=128, num_layers=1, dropout_rate=0.5):
        super(CAT_BiLSTM, self).__init__()
        self.cat = CrossAttention(input_size, dropout_rate, output_size)
        self.bilstm1 = nn.LSTM(output_size, hidden_size, num_layers,bidirectional=True, batch_first=True)
        self.bilstm1 = nn.LSTM(output_size, hidden_size, num_layers,bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X_ag, X_ab):
        ag_output, ab_output = self.cat(X_ag, X_ab)
        output1, _ = self.bilstm1(ag_output)
        output2, _ = self.bilstm1(ab_output)

        output1 = self.dropout(output1)
        output2 = self.dropout(output2)

        result = torch.matmul(output1, output2.transpose(1, 2))

        return result
