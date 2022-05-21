import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import gpytorch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.layer1 = nn.Linear(868, 400)
        self.layer2 = nn.Linear(400, 40)
        # self.layer3 = nn.Linear(40,1)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        # out = self.layer3(out)
        return out

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=40)
        # self.feature_extractor = feature_extractor

    def forward(self, x):
        # z = self.feature_extractor(x)
        # z_normalized = z - z.min(0)[0]
        # z_normalized = 2 * (z_normalized / z_normalized.max(0)[0]) - 1
        # x_normalized = x - x.min(0)[0]
        # x_normalized = 2 * (x_normalized / x_normalized.max(0)[0]) - 1
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        self.covar_x = covar_x
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_covar(self):
        return self.covar_x


class DeepKernelGP():
    """
        Implementation of SIM_DKGP model
    """
    def __init__(self, model, gp_model, sim_dataset, recommd_data, recomend_lr, gp_lr, inner_steps=1, task_example_size=10):
        """
            sim_dataset: [(x1, x2, score)] list of tuples
            recommd_data: recommendation dataset list of [uid, mid, ...]
        """
        # important objects
        # self.tasks = tasks_iterator
        self.dataset = sim_dataset

        self.recommd_data = recommd_data
        self.task_example_size = task_example_size
        self.inner_steps = inner_steps

        self.model = model
        self.gp_model = gp_model
        self.criterion = nn.MSELoss(size_average = True)

        # self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
        # self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        ## GP
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, gp_model)
        # self.optimizer = torch.optim.Adam([{'params': gp_model.parameters(), 'lr': gp_lr},
        #                               {'params': model.parameters(), 'lr': recomend_lr}])
        # self.optimizer_recomd = torch.optim.Adam([{'params': model.parameters(), 'lr': recomend_lr}])
        self.optimizer_gp = torch.optim.Adam([{'params': gp_model.parameters(), 'lr': gp_lr}])

        # hyperparameters
        # self.inner_lr = inner_lr
        # self.meta_lr = meta_lr
        # self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        # self.tasks_per_meta_batch = tasks_per_meta_batch

        #
        self.debug_enable = False

        # metrics
        self.plot_every = 10
        self.print_every = 3
        self.meta_losses = []

    def filter_valid_dataset(self, dataset):
        valid_dataset = []
        for w in dataset:
            if w[0] != -1:
                valid_dataset.append(w)
        return valid_dataset

    def sample_data(self, dataset):
        """ sample example size from dataset
        Returns:
            [(x1, x2, z)]

            for fix input shape, dummpy inputs [-1, -1, 0.0]
        """
        valid_dataset = self.filter_valid_dataset(dataset)
        if len(valid_dataset) < self.task_example_size:
            no_match_cnt = self.task_example_size - len(valid_dataset)
            final_dataset = valid_dataset.extend([(-1, -1, 0.0) for i in range(no_match_cnt)])
            return final_dataset
        else:
            return random.sample(valid_dataset, self.task_example_size)

    def gp_train(self, iteration, task_dataset):
        x1 = [w[0] for w in task_dataset]
        x2 = [w[1] for w in task_dataset]
        target = [w[2] for w in task_dataset]
        x1, x2, target = torch.LongTensor(x1), torch.LongTensor(x2), torch.FloatTensor(target)
        self.optimizer_gp.zero_grad()
        for step in range(self.inner_steps):
            input_data = []
            input_data.extend(self.recommd_data)
            input_data.extend([x1, x2, target])

            # deep net, query shared embedding
            y_logits, sim_input = self.model(input_data)

            # gp
            self.gp_model.set_train_data(inputs=sim_input, targets=target)
            predictions = self.gp_model(sim_input)
            loss = -self.mll(predictions, self.gp_model.train_targets)
            loss.backward()
            self.optimizer_gp.step()
            mse = self.criterion(predictions.mean, target)
            if (iteration % self.print_every == 0 and self.debug_enable):
                print('[%d] Step %d - Deep Kernel Transfer GP Loss: %.3f  MSE: %.3f  lengthscale: %.3f   noise: %.3f' % (
                    iteration, step, loss.item(), mse.item(),
                    0.0,  # gp.covar_module.base_kernel.lengthscale.item(),
                    self.gp_model.likelihood.noise.item()
                ))
        return loss

    def main_loop(self, num_gp_iterations):
        epoch_loss = 0

        ## Training
        self.likelihood.train()
        self.gp_model.train()
        # With updating
        self.model.eval()
        for iteration in range(1, num_gp_iterations + 1):
            task_dataset = self.sample_data(self.dataset)
            if task_dataset is None:
                continue
            if iteration % self.print_every == 0 and self.debug_enable:
                print("DEBUG: DeepKernelGP main loop input dataset size %d, %s" % (len(self.dataset), str(self.dataset[0])))
                print("DEBUG: DeepKernelGP main loop sampled task_dataset size %d" % len(task_dataset))
            loss = self.gp_train(iteration, task_dataset)
            epoch_loss += loss
            if iteration % self.print_every == 0 and self.debug_enable:
                print("Deep Kernel Transfer Main Loop {}/{}. GP loss: {}".format(iteration, num_gp_iterations, epoch_loss / iteration))

class Model_DNN(nn.Module):
    def __init__(self, n_uid, n_mid, n_cat, embedding_size, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False):
        super(Model_DNN, self).__init__()
        self.uid_embed_layer = torch.nn.Embedding(n_uid, embedding_size)
        self.mid_embed_layer = torch.nn.Embedding(n_mid, embedding_size)
        self.cat_embed_layer = torch.nn.Embedding(n_cat, embedding_size)
        # self.fcn = nn.Sequential(
        #     nn.BatchNorm1d(90),
        #     nn.Linear(90, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 80),
        #     nn.ReLU(),
        #     nn.Linear(80, 2))
        self.fcn = nn.Sequential(
            # nn.BatchNorm1d(90),
            nn.Linear(90, 200),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 1))

    def forward(self, input):
        uid_batch_ph, mid_batch_ph, mid_his_batch_ph, cat_batch_ph, cat_his_batch_ph, mask, seq_len_ph, target_ph, lr = input
        self.uid_batch_embedded = self.uid_embed_layer(uid_batch_ph)
        self.mid_batch_embedded = self.mid_embed_layer(mid_batch_ph)
        self.mid_his_batch_embedded = self.mid_embed_layer(mid_his_batch_ph)
        self.cat_batch_embedded = self.cat_embed_layer(cat_batch_ph)
        self.cat_his_batch_embedded = self.cat_embed_layer(cat_his_batch_ph)
        self.item_eb = torch.cat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = torch.cat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = torch.sum(self.item_his_eb, 1)

        inp = torch.cat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        # print ("DEBUG: Model Embedding Layer size %s" % str(inp.shape))
        out = self.fcn(inp)
        # y_hat = nn.Softmax(dim=1)(out)
        return out

class Model_DNN_SIM_NN(nn.Module):
    def __init__(self, n_uid, n_mid, n_cat, embedding_size, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False):
        super(Model_DNN_SIM_NN, self).__init__()
        self.uid_embed_layer = torch.nn.Embedding(n_uid, embedding_size)
        self.mid_embed_layer = torch.nn.Embedding(n_mid, embedding_size)
        self.cat_embed_layer = torch.nn.Embedding(n_cat, embedding_size)
        self.fcn = nn.Sequential(
            # nn.BatchNorm1d(90),
            nn.Linear(90, 200),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 1))
        self.sim_pred_model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(embedding_size, 400)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(400,40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40, 1))
        ]))

    def forward(self, input):
        uid_batch_ph, mid_batch_ph, mid_his_batch_ph, cat_batch_ph, cat_his_batch_ph, mask, seq_len_ph, target_ph, lr, cat_id_1, cat_id_2, sim_target = input
        # shared layers
        self.uid_batch_embedded = self.uid_embed_layer(uid_batch_ph)
        self.mid_batch_embedded = self.mid_embed_layer(mid_batch_ph)
        self.mid_his_batch_embedded = self.mid_embed_layer(mid_his_batch_ph)
        self.cat_batch_embedded = self.cat_embed_layer(cat_batch_ph)
        self.cat_his_batch_embedded = self.cat_embed_layer(cat_his_batch_ph)
        self.item_eb = torch.cat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = torch.cat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = torch.sum(self.item_his_eb, 1)

        ## calcucate similarity, user category embedding only or concatenation of item emb and cat emb
        self.id_1_embedded = self.cat_embed_layer(cat_id_1)
        self.id_2_embedded = self.cat_embed_layer(cat_id_2)

        # task1: recommendation
        inp = torch.cat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        y_logits = self.fcn(inp)

        # task2: similarity prediction, (item_id_1, item_id_2) and (item_id_2, item_id_1) produce same result
        sim_input = self.id_1_embedded + self.id_2_embedded + torch.mul(self.id_1_embedded, self.id_2_embedded)
        z_out = self.sim_pred_model(sim_input) # regression without sigmoid activation
        return y_logits, z_out

class Model_DNN_SIM_SIAMESE(nn.Module):
    def __init__(self, n_uid, n_mid, n_cat, embedding_size, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False):
        super(Model_DNN_SIM_SIAMESE, self).__init__()
        self.uid_embed_layer = torch.nn.Embedding(n_uid, embedding_size)
        self.mid_embed_layer = torch.nn.Embedding(n_mid, embedding_size)
        self.cat_embed_layer = torch.nn.Embedding(n_cat, embedding_size)
        self.embedding_size = embedding_size
        self.fcn = nn.Sequential(
            # nn.BatchNorm1d(90),
            nn.Linear(90, 200),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 1))
        self.sim_pred_model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(embedding_size, 400)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(400,40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40, 1))
        ]))

    def forward(self, input):
        uid_batch_ph, mid_batch_ph, mid_his_batch_ph, cat_batch_ph, cat_his_batch_ph, mask, seq_len_ph, target_ph, lr, cat_id_1, cat_id_2, sim_target = input
        # shared layers
        self.uid_batch_embedded = self.uid_embed_layer(uid_batch_ph)
        self.mid_batch_embedded = self.mid_embed_layer(mid_batch_ph)
        self.mid_his_batch_embedded = self.mid_embed_layer(mid_his_batch_ph)
        self.cat_batch_embedded = self.cat_embed_layer(cat_batch_ph)
        self.cat_his_batch_embedded = self.cat_embed_layer(cat_his_batch_ph)
        self.item_eb = torch.cat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = torch.cat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = torch.sum(self.item_his_eb, 1)

        ## calcucate similarity, user category embedding only or concatenation of item emb and cat emb
        self.id_1_embedded = self.cat_embed_layer(cat_id_1)
        self.id_2_embedded = self.cat_embed_layer(cat_id_2)

        # task1: recommendation
        inp = torch.cat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        y_logits = self.fcn(inp)

        # task2: siamese network, weight on L1 norm
        l1_norm = torch.abs(self.id_1_embedded - self.id_2_embedded) # l1_norm most similar 1
        sim = 1 - l1_norm
        z_out = nn.Linear(self.embedding_size, 1)(sim)
        # z_out = nn.Sigmoid()(nn.Linear(self.embedding_size, 1)(l1_norm)) # regression to [0,1]
        return y_logits, z_out

class Model_DNN_SIM_DeepKernelGP(nn.Module):
    """
        MAML optimized model with similarity prediction task
    """
    def __init__(self, n_uid, n_mid, n_cat, embedding_size, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False):
        super(Model_DNN_SIM_DeepKernelGP, self).__init__()
        self.uid_embed_layer = torch.nn.Embedding(n_uid, embedding_size)
        self.mid_embed_layer = torch.nn.Embedding(n_mid, embedding_size)
        self.cat_embed_layer = torch.nn.Embedding(n_cat, embedding_size)
        self.fcn = nn.Sequential(
            # nn.BatchNorm1d(90),
            nn.Linear(90, 200),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 1))

    def parameterised(self, input_data, weights):
        """
        Args:
            weight: updated weights to calculate forward pass

        Returns:
        """
        #print ("DEBUG: Model_DNN_SIM_MAML input_data %s" % str(input_data))
        #print ("DEBUG: Model_DNN_SIM_MAML weight %s" % str(weights))
        #for i, w in enumerate(weights):
        #    print ("DEBUG: Model_DNN_SIM_MAML weight %d,shape %s" % (i, str(w.size() if type(w) is not float else "0")))

        uid_batch_ph, mid_batch_ph, mid_his_batch_ph, cat_batch_ph, cat_his_batch_ph, mask, seq_len_ph, target_ph, lr, cat_id_1, cat_id_2, sim_target = input_data
        self.id_1_embedded = self.cat_embed_layer(cat_id_1)
        self.id_2_embedded = self.cat_embed_layer(cat_id_2)
        sim_input = self.id_1_embedded + self.id_2_embedded + torch.mul(self.id_1_embedded, self.id_2_embedded)

        # sim network
        return None, sim_input

    def forward(self, input):
        uid_batch_ph, mid_batch_ph, mid_his_batch_ph, cat_batch_ph, cat_his_batch_ph, mask, seq_len_ph, target_ph, lr, cat_id_1, cat_id_2, sim_target = input
        # shared layers
        self.uid_batch_embedded = self.uid_embed_layer(uid_batch_ph)
        self.mid_batch_embedded = self.mid_embed_layer(mid_batch_ph)
        self.mid_his_batch_embedded = self.mid_embed_layer(mid_his_batch_ph)
        self.cat_batch_embedded = self.cat_embed_layer(cat_batch_ph)
        self.cat_his_batch_embedded = self.cat_embed_layer(cat_his_batch_ph)
        self.item_eb = torch.cat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = torch.cat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = torch.sum(self.item_his_eb, 1)

        # task1: recommendation
        inp = torch.cat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        y_logits = self.fcn(inp)

        # task2: similarity prediction input to deep GP model, (item_id_1, item_id_2) and (item_id_2, item_id_1) produce same result
        self.id_1_embedded = self.cat_embed_layer(cat_id_1)
        self.id_2_embedded = self.cat_embed_layer(cat_id_2)
        sim_input = self.id_1_embedded + self.id_2_embedded + torch.mul(self.id_1_embedded, self.id_2_embedded)
        return y_logits, sim_input

# Base Similarity Prediction NN Module
class Model_SIM_PRED_DNN(nn.Module):
    def __init__(self, embedding_size):
        super(Model_SIM_PRED_DNN, self).__init__()
        self.sim_pred_model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(embedding_size, 40)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(40, 1))
        ]))

    def parameterised(self, sim_input, weights):
        x = nn.functional.linear(sim_input, weights[0], weights[1])
        x = nn.functional.relu(x)
        z_out = nn.functional.linear(x, weights[2], weights[3])
        return z_out

    def forward(self, sim_input):
        out = self.sim_pred_model(sim_input)
        return out

class Model_DNN_SIM_MAML(nn.Module):
    """
        MAML optimized model with similarity prediction task
    """
    def __init__(self, n_uid, n_mid, n_cat, embedding_size, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False):
        super(Model_DNN_SIM_MAML, self).__init__()
        self.uid_embed_layer = torch.nn.Embedding(n_uid, embedding_size)
        self.mid_embed_layer = torch.nn.Embedding(n_mid, embedding_size)
        self.cat_embed_layer = torch.nn.Embedding(n_cat, embedding_size)
        self.fcn = nn.Sequential(
            # nn.BatchNorm1d(90),
            nn.Linear(90, 200),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 1))
        # self.sim_pred_model = nn.Sequential(OrderedDict([
        #     ('l1', nn.Linear(embedding_size, 400)),
        #     ('relu1', nn.ReLU()),
        #     ('l2', nn.Linear(400,40)),
        #     ('relu2', nn.ReLU()),
        #     ('l3', nn.Linear(40, 1))
        # ]))

    def parameterised(self, input_data, weights):
        """
        Args:
            weight: updated weights to calculate forward pass

        Returns:
        """
        #print ("DEBUG: Model_DNN_SIM_MAML input_data %s" % str(input_data))
        #print ("DEBUG: Model_DNN_SIM_MAML weight %s" % str(weights))
        #for i, w in enumerate(weights):
        #    print ("DEBUG: Model_DNN_SIM_MAML weight %d,shape %s" % (i, str(w.size() if type(w) is not float else "0")))

        uid_batch_ph, mid_batch_ph, mid_his_batch_ph, cat_batch_ph, cat_his_batch_ph, mask, seq_len_ph, target_ph, lr, cat_id_1, cat_id_2, sim_target = input_data
        self.id_1_embedded = self.cat_embed_layer(cat_id_1)
        self.id_2_embedded = self.cat_embed_layer(cat_id_2)
        sim_input = self.id_1_embedded + self.id_2_embedded + torch.mul(self.id_1_embedded, self.id_2_embedded)

        # # sim network
        # x = nn.functional.linear(x, weights[9], weights[10])
        # x = nn.functional.relu(x)
        # x = nn.functional.linear(x, weights[11], weights[12])
        # x = nn.functional.relu(x)
        # z_out = nn.functional.linear(x, weights[13], weights[14])
        return None, sim_input

    def forward(self, input):
        uid_batch_ph, mid_batch_ph, mid_his_batch_ph, cat_batch_ph, cat_his_batch_ph, mask, seq_len_ph, target_ph, lr, cat_id_1, cat_id_2, sim_target = input
        # shared layers
        self.uid_batch_embedded = self.uid_embed_layer(uid_batch_ph)
        self.mid_batch_embedded = self.mid_embed_layer(mid_batch_ph)
        self.mid_his_batch_embedded = self.mid_embed_layer(mid_his_batch_ph)
        self.cat_batch_embedded = self.cat_embed_layer(cat_batch_ph)
        self.cat_his_batch_embedded = self.cat_embed_layer(cat_his_batch_ph)
        self.item_eb = torch.cat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = torch.cat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = torch.sum(self.item_his_eb, 1)

        self.id_1_embedded = self.cat_embed_layer(cat_id_1)
        self.id_2_embedded = self.cat_embed_layer(cat_id_2)

        # task1: recommendation
        inp = torch.cat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        y_logits = self.fcn(inp)

        # task2: similarity prediction, (item_id_1, item_id_2) and (item_id_2, item_id_1) produce same result
        sim_input = self.id_1_embedded + self.id_2_embedded + torch.mul(self.id_1_embedded, self.id_2_embedded)
        # z_out = self.sim_pred_model(sim_input)
        return y_logits, sim_input

class Model_DNN_SIM_MAML_v1(nn.Module):
    """
        MAML optimized model with similarity prediction task
    """
    def __init__(self, n_uid, n_mid, n_cat, embedding_size, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False):
        super(Model_DNN_SIM_MAML, self).__init__()
        self.uid_embed_layer = torch.nn.Embedding(n_uid, embedding_size)
        self.mid_embed_layer = torch.nn.Embedding(n_mid, embedding_size)
        self.cat_embed_layer = torch.nn.Embedding(n_cat, embedding_size)
        self.fcn = nn.Sequential(
            # nn.BatchNorm1d(90),
            nn.Linear(90, 200),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 1))
        self.sim_pred_model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(embedding_size, 400)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(400,40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40, 1))
        ]))

    def parameterised(self, input_data, weights):
        """
        Args:
            weight: updated weights to calculate forward pass

        Returns:
        """
        #print ("DEBUG: Model_DNN_SIM_MAML input_data %s" % str(input_data))
        #print ("DEBUG: Model_DNN_SIM_MAML weight %s" % str(weights))
        #for i, w in enumerate(weights):
        #    print ("DEBUG: Model_DNN_SIM_MAML weight %d,shape %s" % (i, str(w.size() if type(w) is not float else "0")))

        uid_batch_ph, mid_batch_ph, mid_his_batch_ph, cat_batch_ph, cat_his_batch_ph, mask, seq_len_ph, target_ph, lr, cat_id_1, cat_id_2, sim_target = input_data
        self.id_1_embedded = self.cat_embed_layer(cat_id_1)
        self.id_2_embedded = self.cat_embed_layer(cat_id_2)
        x = self.id_1_embedded + self.id_2_embedded + torch.mul(self.id_1_embedded, self.id_2_embedded)

        # sim network
        x = nn.functional.linear(x, weights[9], weights[10])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[11], weights[12])
        x = nn.functional.relu(x)
        z_out = nn.functional.linear(x, weights[13], weights[14])
        return None, z_out

    def forward(self, input):
        uid_batch_ph, mid_batch_ph, mid_his_batch_ph, cat_batch_ph, cat_his_batch_ph, mask, seq_len_ph, target_ph, lr, cat_id_1, cat_id_2, sim_target = input
        # shared layers
        self.uid_batch_embedded = self.uid_embed_layer(uid_batch_ph)
        self.mid_batch_embedded = self.mid_embed_layer(mid_batch_ph)
        self.mid_his_batch_embedded = self.mid_embed_layer(mid_his_batch_ph)
        self.cat_batch_embedded = self.cat_embed_layer(cat_batch_ph)
        self.cat_his_batch_embedded = self.cat_embed_layer(cat_his_batch_ph)
        self.item_eb = torch.cat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = torch.cat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = torch.sum(self.item_his_eb, 1)

        self.id_1_embedded = self.cat_embed_layer(cat_id_1)
        self.id_2_embedded = self.cat_embed_layer(cat_id_2)

        # task1: recommendation
        inp = torch.cat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        y_logits = self.fcn(inp)

        # task2: similarity prediction, (item_id_1, item_id_2) and (item_id_2, item_id_1) produce same result
        sim_input = self.id_1_embedded + self.id_2_embedded + torch.mul(self.id_1_embedded, self.id_2_embedded)
        z_out = self.sim_pred_model(sim_input)
        return y_logits, z_out


class MAML():
    def __init__(self, model, sim_pred_model, sim_dataset, recommd_data, inner_lr, meta_lr, inner_steps=1, task_example_size=10, tasks_per_meta_batch=1000):
        """
            sim_dataset: [(x1, x2, score)] list of tuples
            recommd_data: recommendation dataset list of [uid, mid, ...]
        """
        # important objects
        # self.tasks = tasks_iterator
        self.task_example_size = task_example_size
        self.dataset = sim_dataset
        self.recommd_data = recommd_data

        self.model = model
        self.sim_pred_model = sim_pred_model
        self.weights = list(self.sim_pred_model.parameters())  # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss(size_average = True)
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        self.tasks_per_meta_batch = tasks_per_meta_batch

        #
        self.debug_enable = False

        # metrics
        self.plot_every = 10
        self.print_every = 10
        self.meta_losses = []

    def sample_data(self, dataset):
        """ sample example size from dataset
        Returns:
            [(x1, x2, z)]
        """
        if len(dataset) < self.task_example_size:
            return dataset
        else:
            return random.sample(dataset, self.task_example_size)

    def inner_loop(self, task_dataset):
        x1 = [w[0] for w in task_dataset]
        x2 = [w[1] for w in task_dataset]
        target = [w[2] for w in task_dataset]

        x1, x2, target = torch.LongTensor(x1), torch.LongTensor(x2), torch.FloatTensor(target)

        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.sim_pred_model.parameters()]

        # perform training on data sampled from task
        for step in range(self.inner_steps):
            input_data = []
            input_data.extend(self.recommd_data)
            input_data.extend([x1, x2, target])
            # recommendation module
            y_logits, sim_input = self.model(input_data)
            # sim_predict module
            z_out = self.sim_pred_model(sim_input)
            loss = self.criterion(z_out, target)
            # print ("DEBUG: INNER_LOOP processing step %d, %f" % (step, loss))
            # print ("DEBUG: INNER_LOOP processing step %d, temp_weights %s" % (step, str(temp_weights)))
            # compute grad and update inner loop weights
            # grad = torch.autograd.grad(loss, temp_weights)
            loss.backward()
            # print ("DEBUG: INNER_LOOP processing step %d, model parameters %s" % (step, str(self.model.parameters())))
            # print ("DEBUG: INNER_LOOP processing step %d, temp_weights %s" % (step, str(temp_weights)))
            grad = [w.grad for w in self.sim_pred_model.parameters()]
            #grad = torch.autograd.grad(loss, self.model.parameters())
            # print ("DEBUG: INNER_LOOP processing step %d, grad %s" % (step, str(grad)))
            temp_weights = [w - self.inner_lr * g if g is not None else 0.0 for w, g in zip(temp_weights, grad)]
            break

        # sample new data for meta-update and compute loss
        # computed loss on updated parameters
        sample_dataset = self.sample_data(self.dataset)
        x1 = [w[0] for w in sample_dataset]
        x2 = [w[1] for w in sample_dataset]
        target = [w[2] for w in sample_dataset]
        x1, x2, target = torch.LongTensor(x1), torch.LongTensor(x2), torch.FloatTensor(target)
        input_data = []
        input_data.extend(self.recommd_data)
        input_data.extend([x1, x2, target])

        y_logits, sim_input = self.model(input_data)
        z_out = self.sim_pred_model.parameterised(sim_input, temp_weights)

        # y_logits, z_out = self.model.parameterised(input_data, temp_weights)
        loss = self.criterion(z_out, target)
        if self.debug_enable:
            print ("DEBUG: INNER_LOOP processing final loss %f" % (loss))
        return loss

    def main_loop(self, num_iterations):
        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):

            # compute meta loss
            meta_loss = 0
            for i in range(self.tasks_per_meta_batch):
                task_dataset = self.sample_data(self.dataset)
                meta_loss += self.inner_loop(task_dataset)
                if i % 10 == 0 and self.debug_enable:
                    print ("DEBUG: Main Loop Cum Meta Loss at meta_batch %d, is %f" % (i, meta_loss))

            # compute meta gradient of loss with respect to maml weights
            # meta_grads = torch.autograd.grad(meta_loss, self.weights)

            meta_loss.backward()
            if self.debug_enable:
                print ("DEBUG: main_loop processing step meta_loss %s" % str(meta_loss))
            meta_grads = [w.grad for w in self.sim_pred_model.parameters()]
            # print ("DEBUG: main_loop processing step meta_grads %s" % str(meta_grads))

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # log metrics
            epoch_loss += meta_loss.item() / self.tasks_per_meta_batch

            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))

            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0

class MAML_v1():
    def __init__(self, model, sim_dataset, recommd_data, inner_lr, meta_lr, inner_steps=1, task_example_size=10, tasks_per_meta_batch=1000):
        """
            sim_dataset: [(x1, x2, score)] list of tuples
            recommd_data: recommendation dataset list of [uid, mid, ...]
        """
        # important objects
        # self.tasks = tasks_iterator
        self.task_example_size = task_example_size
        self.dataset = sim_dataset
        self.recommd_data = recommd_data

        self.model = model
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss(size_average = True)
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        self.tasks_per_meta_batch = tasks_per_meta_batch

        #
        self.debug_enable = False

        # metrics
        self.plot_every = 10
        self.print_every = 10
        self.meta_losses = []

    def sample_data(self, dataset):
        """ sample example size from dataset
        Returns:
            [(x1, x2, z)]
        """
        if len(dataset) < self.task_example_size:
            return dataset
        else:
            return random.sample(dataset, self.task_example_size)

    def inner_loop(self, task_dataset):
        x1 = [w[0] for w in task_dataset]
        x2 = [w[1] for w in task_dataset]
        target = [w[2] for w in task_dataset]

        x1, x2, target = torch.LongTensor(x1), torch.LongTensor(x2), torch.FloatTensor(target)

        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.model.parameters()]

        # perform training on data sampled from task
        for step in range(self.inner_steps):
            input_data = []
            input_data.extend(self.recommd_data)
            input_data.extend([x1, x2, target])
            y_logits, z_out = self.model(input_data)
            # print ("DEBUG: INNER_LOOP processing step %d, shape %s, z_out %s" % (step, str(z_out.size()), str(z_out)))
            # print ("DEBUG: INNER_LOOP processing step %d, shape %s, target %s" % (step, str(target.size()), str(target)))

            loss = self.criterion(z_out, target)
            # print ("DEBUG: INNER_LOOP processing step %d, %f" % (step, loss))
            # print ("DEBUG: INNER_LOOP processing step %d, temp_weights %s" % (step, str(temp_weights)))
            # compute grad and update inner loop weights
            # grad = torch.autograd.grad(loss, temp_weights)
            loss.backward()
            # print ("DEBUG: INNER_LOOP processing step %d, model parameters %s" % (step, str(self.model.parameters())))
            # print ("DEBUG: INNER_LOOP processing step %d, temp_weights %s" % (step, str(temp_weights)))

            grad = [w.grad for w in self.model.parameters()]
            #grad = torch.autograd.grad(loss, self.model.parameters())
            # print ("DEBUG: INNER_LOOP processing step %d, grad %s" % (step, str(grad)))
            temp_weights = [w - self.inner_lr * g if g is not None else 0.0 for w, g in zip(temp_weights, grad)]
            break

        # sample new data for meta-update and compute loss
        # computed loss on updated parameters
        sample_dataset = self.sample_data(self.dataset)
        x1 = [w[0] for w in sample_dataset]
        x2 = [w[1] for w in sample_dataset]
        target = [w[2] for w in sample_dataset]
        x1, x2, target = torch.LongTensor(x1), torch.LongTensor(x2), torch.FloatTensor(target)
        input_data = []
        input_data.extend(self.recommd_data)
        input_data.extend([x1, x2, target])
        y_logits, z_out = self.model.parameterised(input_data, temp_weights)
        loss = self.criterion(z_out, target)
        if self.debug_enable:
            print ("DEBUG: INNER_LOOP processing final loss %f" % (loss))
        return loss

    def main_loop(self, num_iterations):
        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):

            # compute meta loss
            meta_loss = 0
            for i in range(self.tasks_per_meta_batch):
                task_dataset = self.sample_data(self.dataset)
                meta_loss += self.inner_loop(task_dataset)
                if i % 10 == 0 and self.debug_enable:
                    print ("DEBUG: Main Loop Cum Meta Loss at meta_batch %d, is %f" % (i, meta_loss))

            # compute meta gradient of loss with respect to maml weights
            # meta_grads = torch.autograd.grad(meta_loss, self.weights)

            meta_loss.backward()
            if self.debug_enable:
                print ("DEBUG: main_loop processing step meta_loss %s" % str(meta_loss))
            meta_grads = [w.grad for w in self.model.parameters()]
            # print ("DEBUG: main_loop processing step meta_grads %s" % str(meta_grads))

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # log metrics
            epoch_loss += meta_loss.item() / self.tasks_per_meta_batch

            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))

            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0
