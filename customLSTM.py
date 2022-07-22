class MYLSTM(nn.Module):
    def __init__(self, D_in, H, D_out, num_layers=1, bidirectional=false):
        super(MYLSTM, self).__init__()
        self.hidden_dim = H
        self.num_layers = num_layers
        self.num_direction =  2 if bidirectional else 1
        self.lstm = nn.LSTM(D_in, H, num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden2out = nn.Linear(self.num_direction*self.hidden_dim, D_out)

    def init_hidden(self, batch_size):
        #here some parameters should be defult some of them should change
        h, c = (Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)))
        return h,c

    def forward(self, sequence, lengths, h, c):
        sequence = nn.utils.lstm.pack_padded_sequence(sequence, lengths, batch_first=True)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = nn.utils.lstm.pad_packed_sequence(output, batch_first=True)
        output = self.hidden2out(output)
        return output