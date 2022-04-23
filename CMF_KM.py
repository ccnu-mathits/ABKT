import torch
from DatasetLoading import get_split_sequences
import pickle
import os
import numpy as np
import torch.utils.data as Data
import torch.optim as opt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score
from PytorchModels import K_CMF,IRT_2

class CMF:
    def __init__(self,
                 dataset='ASSISTment2009',
                 type = 'RandomIterateSection',
                 min_length=10,
                 early_stop=10,
                 epoch=100,
                 m_lr=0.0005,
                 k_hidden_size = 5,
                 device = 'cpu',
                 guess = 0.25,
                 ):

        self.dataset = dataset
        self.type = type
        self.k_hidden_size = k_hidden_size
        self.early_stop = early_stop
        self.epoch = epoch
        self.m_lr = m_lr
        self.device = device
        self.guess = guess

        save_dataset_path = './ProcessedData/' + dataset + '-' + type + '-' + str(min_length) +'-squence'
        if os.access(save_dataset_path, os.F_OK):
            print("Processed data is existent...")
            print('Loading...')
            save_dataset_file = open(save_dataset_path, 'rb')
            [self.user_num,self.item_num,self.skill_num,self.record_num,train_sequences,test_triplet,Q_matrix_s]\
                = pickle.load(save_dataset_file)
            save_dataset_file.close()
        else:
            print("Processed data is not existent...")
            self.user_num,self.item_num,self.skill_num,self.record_num,train_sequences,test_triplet,Q_matrix_s = \
                get_split_sequences(dataset, type, min_length)

            train_test_sets = [self.user_num,self.item_num,self.skill_num,
                               self.record_num,train_sequences,test_triplet,Q_matrix_s]
            save_dataset_file = open(save_dataset_path, 'wb')
            pickle.dump(train_test_sets, save_dataset_file)
            save_dataset_file.close()
            print('Training set and test set are saved in', save_dataset_path)
        print("Data is processed which has:", self.user_num, 'students,',
                  self.item_num, 'questions,',
                  self.skill_num, 'skills,', self.record_num, 'records')

        print('*'*20,'hyperparameters','*'*20)
        print('k_hidden_num:',self.k_hidden_size)
        print('guess:',self.guess)

        # 将数据整理放入torch中
        # 划分方法为每一个序列中随机分为Train和Test，其中序列的第一个不能是Test
        # train_squences {userid:[[squence(itemid)],[squence(correct)]]}
        # test_triplet [[userid,itemid,corect],...]

        Q_matrix = torch.zeros((self.item_num, self.skill_num))
        index = 0
        for i in Q_matrix_s:
            for ii in i:
                Q_matrix[index][int(ii)] = 1
            index += 1
        self.Q_matrix = Q_matrix.to(self.device)

        self.train_users = []
        self.train_itemsq = []
        self.train_correctsq = []
        self.train_itemsq_length = []
        for user in train_sequences:
            self.train_users.append(user)
            self.train_itemsq.append(torch.tensor(train_sequences[user][0]).squeeze(0))
            self.train_correctsq.append(torch.tensor(train_sequences[user][1]).squeeze(0).long())
            self.train_itemsq_length.append(train_sequences[user][0].__len__())
        self.train_itemsq_length = torch.tensor(self.train_itemsq_length)

        self.test_sets = torch.tensor(test_triplet).long().to(self.device)
        self.test_users = list(set(self.test_sets[:,0].tolist()))

        # 训练的index
        self.train_index = torch.arange(0, self.train_users.__len__(), 1)


    def train(self):
        print('*' * 20, 'start training', '*' * 20)
        train_data_loader = Data.DataLoader(
            dataset=self.train_index,
            batch_size=1,
            shuffle=True
        )
        self.K_CMF = K_CMF(
            self.k_hidden_size,
            self.skill_num,
            self.user_num,
            self.item_num,
            self.Q_matrix,
        ).to(self.device)
        BCEloss = torch.nn.BCELoss()

        train_vars = list(self.K_CMF.parameters())
        optimizer = opt.Adam(train_vars, lr=self.m_lr)

        self.bestACC = 0
        self.bestAUC = 0
        stop = 0

        for epoch in range(self.epoch):

            self.K_CMF.train()

            bce_loss = 0
            train_pred_all = []
            train_pred01_all = []
            train_correct_all = []
            # for ii,index in enumerate(train_data_loader):
            # for index in tqdm(train_data_loader):
            for index in train_data_loader:
                user = self.train_users[index]
                itemsq = self.train_itemsq[index].to(self.device)
                correctsq = self.train_correctsq[index].to(self.device)
                itemsq_length = self.train_itemsq_length[index]
                itemsq.to(self.device)
                user_k,_,_ = self.K_CMF.forward(user,itemsq)
                item_q = self.Q_matrix[itemsq, :]
                item_k = self.K_CMF.item_k[itemsq,:]

                pred = IRT_2(user_k[:-1, :],item_k,item_q,self.guess)

                train_pred_all.append(pred.detach().cpu().numpy())
                train_pred01_all.append(pred.ge(0.5).float().detach().cpu().numpy())
                train_correct_all.append(correctsq.cpu().numpy())
                loss = BCEloss(pred.clamp(0,1), correctsq.float())
                bce_loss += loss
                total_loss = loss
                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer.step()

            train_y_all = np.hstack(train_correct_all)
            train_pred_all = np.hstack(train_pred_all)
            train_pred01_all = np.hstack(train_pred01_all)

            bce_loss /= self.train_index.shape[0]
            train_ACC = accuracy_score(train_y_all, train_pred01_all)
            train_AUC = roc_auc_score(train_y_all, train_pred_all)

            print('Epoch:', epoch, '| BCEloss:', bce_loss.cpu().detach().numpy(), '| ACC:', train_ACC, '| AUC:',
                  train_AUC)

            self.K_CMF.eval()

            test_user_state_k = []
            # get the state of each user
            for test_index in range(self.test_users.__len__()):
                test_user = self.test_users[test_index]
                train_index = self.train_users.index(test_user)
                train_index_itemsq = self.train_itemsq[train_index].to(self.device)
                train_model_output_k,_,_ = self.K_CMF.forward(train_index,train_index_itemsq)
                test_out_k = train_model_output_k[-1,:]
                test_user_state_k.append(test_out_k.unsqueeze(0))

            test_user_state_k = torch.cat(test_user_state_k,0)

            user_states_k = test_user_state_k[self.test_sets[:,0],:]
            item_states_q = self.Q_matrix[self.test_sets[:,1],:]
            item_state_k = self.K_CMF.item_k[self.test_sets[:,1],:]
            pred = IRT_2(user_states_k,item_state_k,item_states_q,self.guess)
            # print(pred.detach().cpu().numpy())

            test_pred_collect = pred.detach().cpu().numpy()
            test_pred01_collect = pred.ge(0.5).float().detach().cpu().numpy()
            test_y_collect = self.test_sets[:,2].cpu().numpy()

            test_pred_all = test_pred_collect
            test_pred01_all = test_pred01_collect
            test_y_all = test_y_collect
            ACC = accuracy_score(test_y_all, test_pred01_all)
            AUC = roc_auc_score(test_y_all, test_pred_all)
            self.ACC = ACC
            self.AUC = AUC

            if AUC > self.bestAUC:
                self.bestAUC = AUC
                save_path_k = './Models/'+str(self.dataset)+'-'+str(self.type)+'/CMF-k-' + str(self.k_hidden_size)+ '-' + str(self.guess) + '-earlystop'
                torch.save(self.K_CMF.state_dict(), save_path_k)
                stop = 0
            if ACC > self.bestACC:
                self.bestACC = ACC
                stop = 0
            else:
                stop = stop + 1
            print('Test ACC:',ACC,'| Test AUC:',AUC)
            print('Best ACC:', self.bestACC, '| Best AUC:', self.bestAUC)
            if stop >= self.early_stop or epoch == self.epoch-1:
                print('*' * 20, 'stop training', '*' * 20)
                save_path_k = './Models/'+str(self.dataset)+'-'+str(self.type)+'/CMF-k-' + str(self.k_hidden_size) + '-' + str(self.guess) + '-epoch' + str(epoch)
                torch.save(self.K_CMF.state_dict(), save_path_k)
                break


    def log_result(self):
        filename = os.path.split(__file__)[-1].split(".")[0]
        f = open("./Results/" + filename + "-" + self.dataset + ".txt", "a+")
        f.write("datasets = " + self.dataset+ "\n")
        f.write("type = " + self.type+ "\n")
        f.write("k_hidden_num = " + str(self.k_hidden_size)+ " guess = " + str(self.guess)+ "\n")
        f.write("Best ACC = " + str(self.bestACC) + "\n")
        f.write("Best AUC = " + str(self.bestAUC) + "\n")
        f.write("Final ACC = " + str(self.ACC) + "\n")
        f.write("Final AUC = " + str(self.AUC) + "\n")
        f.write("\n")
        f.write("\n")
        print("The results are logged!!!")




if __name__ == '__main__':
    hidden_num_range = [5]
    guess_range = [0.25]

    for hr in hidden_num_range:
        for g in guess_range:
            cmf = CMF(m_lr=0.001,
                      early_stop=10,
                      epoch=50,
                      device='cuda:0',
                      k_hidden_size=hr,
                      guess=g,
                      )
            cmf.train()
            cmf.log_result()






