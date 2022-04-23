import torch


def IRT_2(user_k,item_k,item_q,guess):
    d = 1.702
    r = torch.sum((user_k - item_k)*item_q, 1)/torch.sum(item_q,1)
    p = guess + (1-guess)/(1+torch.exp(-d*r))
    return p

class K_CMF(torch.nn.Module):
    def __init__(self,k_hidden_size,skill_num,user_num,item_num,Q_matrix):
        super(K_CMF, self).__init__()
        print('*'*20,'Parameters:','*'*20)
        self.Q_matrix_m = Q_matrix.unsqueeze(2).repeat(1,1,k_hidden_size)
        self.user_initial_k = torch.nn.Parameter(torch.zeros((user_num,skill_num))*0.01)
        print('user_initial_k:',self.user_initial_k.shape)
        self.item_k = torch.nn.Parameter(torch.rand((item_num, skill_num)) * 0.01)
        print('item_k:', self.item_k.shape)
        self.user_improving_k = torch.nn.Parameter(torch.ones((user_num,skill_num,k_hidden_size))*0.01)
        print('user_improving_k:', self.user_improving_k.shape)
        self.item_improving_k = torch.nn.Parameter(torch.ones((item_num,skill_num,k_hidden_size))*0.01)
        print('item_improving_k:', self.item_improving_k.shape)
        self.item_improving_k.data = self.item_improving_k*self.Q_matrix_m.cpu()
        print('*' * 20, 'Parameters:', '*' * 20)

    def forward(self, user, sq):
        length = sq.__len__()
        temp_k = self.user_initial_k[user,:]
        user_improving_k_sq = self.user_improving_k[user,:].unsqueeze(0).repeat(length,1,1)
        sequence_k = []
        sequence_k.append(temp_k.unsqueeze(0))
        item_improving_k_sq = self.item_improving_k[sq, :]
        improves = torch.sum(user_improving_k_sq * item_improving_k_sq, 2)
        # 非负约束
        improves_k = torch.relu(improves)
        for i in range(sq.__len__()):
            improve_k = improves_k[i,:]
            temp_k = temp_k + improve_k
            sequence_k.append(temp_k.unsqueeze(0))
        out = torch.sigmoid(torch.cat(sequence_k,0))
        # out [sq_length+1, skill_num]
        # improve_norm_k = torch.mean(torch.pow(user_improving_k_sq, 2)) + torch.mean(torch.pow(item_improving_k_sq, 2))
        ui_norm_k = 0
        return out,0,0


class GMF(torch.nn.Module):
    def __init__(self,n_users,n_items,embedding_k,aj_norm,adj,layer):
        super(GMF,self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_k = embedding_k
        self.adj_tag = adj
        self.layer = layer
        aj_norm = aj_norm.to(torch.float32)

        self.embeddings = torch.nn.Parameter(torch.randn((n_users+n_items, embedding_k)) * 0.01)
        if layer >= 1:
            self.aj_norm_1 = aj_norm.to_dense()
        if layer >= 2:
            self.aj_norm_2 = self.aj_norm_1.mm(aj_norm.to_dense())
        if layer >= 3 :
            self.aj_norm_3 = self.aj_norm_2.mm(aj_norm.to_dense())

        if adj:
            self.adj = torch.nn.Parameter(torch.ones(aj_norm.shape))
        self.user_GE = torch.nn.Parameter(torch.zeros((n_users)))
        self.item_GE = torch.nn.Parameter(torch.zeros((n_items)))


    def forward(self,user_index,item_index):
        if self.adj_tag:
            if self.layer == 0:
                G_embeddings = self.embeddings
            elif self.layer == 1:
                G_embeddings = (self.aj_norm_1 * self.adj).mm(self.embeddings)
            elif self.layer == 2:
                G_embeddings = (self.aj_norm_2 * self.adj).mm(self.embeddings)
            elif self.layer == 3:
                G_embeddings = (self.aj_norm_3 * self.adj).mm(self.embeddings)
            else:
                G_embeddings = []
                print("Wrong GMF layer setting!!! Choosing From [0,1,2,3]")
        else:
            if self.layer == 0:
                G_embeddings = self.embeddings
            elif self.layer == 1:
                G_embeddings = self.aj_norm_1.mm(self.embeddings)
            elif self.layer == 2:
                G_embeddings = self.aj_norm_2.mm(self.embeddings)
            elif self.layer == 3:
                G_embeddings = self.aj_norm_3.mm(self.embeddings)
            else:
                G_embeddings = []
                print("Wrong GMF layer setting!!! Choosing From [0,1,2,3]")

        user_embeddings_batch = G_embeddings[user_index,:]
        user_GE_batch = self.user_GE[user_index]

        item_embeddings_batch = G_embeddings[item_index+self.n_users, :]
        item_GE_batch = self.item_GE[item_index]

        pred_batch = (user_embeddings_batch*item_embeddings_batch).sum(1) + user_GE_batch + item_GE_batch
        u_norm = torch.mean(torch.pow(user_embeddings_batch, 2))
        i_norm = torch.mean(torch.pow(item_embeddings_batch, 2))
        return pred_batch, u_norm, i_norm



class LSTM_model(torch.nn.Module):
    def __init__(self,item_representations,HIDDEN_SIZE,NUM_LAYER,DROPOUT,SKILL_NUM,input_train):
        super(LSTM_model, self).__init__()

        if input_train:
            self.item_representations = torch.nn.Parameter(item_representations)
        else:
            self.item_representations = item_representations

        self.rnn = torch.nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=item_representations.shape[1],
            hidden_size=HIDDEN_SIZE,  # rnn hidden unit
            num_layers=NUM_LAYER,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.out = torch.nn.Linear(HIDDEN_SIZE, SKILL_NUM)

    def forward(self, sq):
        # sq shape (time_step)
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = self.item_representations[sq,:].unsqueeze(0)
        r_out, _ = self.rnn(x)  # LSTM
        out1 = self.dropout(self.out(r_out))
        # r_out, h_n = self.rnn(x)   # GRU
        out = torch.sigmoid(out1)
        return out