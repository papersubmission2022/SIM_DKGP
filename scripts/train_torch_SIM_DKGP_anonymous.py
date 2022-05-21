import numpy
import math
from model_torch import *
import time
import random
import sys
from similarity import similarity_utils
from similarity.sim_data_iterator import SimDatasetInterator, SimDatasetMetaInterator
from similarity.sim_data_iterator import DataIterator, UserGroupDataIterator
from similarity.sim_data_iterator import read_file_lines
from similarity.sim_data_iterator import load_dict, id_to_index_dict

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

def train_recomd_sim_jointly(
        train_file = "/${your_train_folder}/local_train_splitByUser",
        test_file = "/${your_train_folder}/jointed-negative-sampling-candidates-20",
        sim_train_path="/${your_train_folder}/sim_dataset_Electronics_train.txt",
        sim_test_path="/${your_train_folder}/sim_dataset_Electronics_test.txt",
        uid_voc = "/${your_train_folder}/uid_voc.pkl",
        mid_voc = "/${your_train_folder}/mid_voc.pkl",
        cat_voc = "/${your_train_folder}/cat_voc.pkl",
        item_info_path = "/${your_train_folder}/item-info",
        review_info_path = "/${your_train_folder}/reviews-info",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        diversity_theta = 1.0,
        l_shots = 10,
        user_group_batch_size=20,
        diversify_method = "MMR"
):
    """ Notes: The data iterator and other util functions contains non-anonymous information, these files are not uploaded at the time of review process due to anonymous restriction...
    """
    ## hyperparameters
    total_train_iter = 4
    lr = 0.001
    log_train_iter = 100
    log_test_iter = 100
    num_iterations = 3
    w2 = 0.01
    similarity_confidence_interval_enable = False
    alpha = 1.96 # normal dist. at 0.05 confidence interval

    print("----------- Start Training Stage-----------")
    print("Training total_train_iter:%d" % total_train_iter)
    print("Training l_shots:%d" % l_shots)
    print("Training lr:%f" % lr)
    print("Training model_type:%s" % model_type)
    print("Training num_iterations inner loop:%d" % num_iterations)
    print("Testing Diversity Theta:%f" % diversity_theta)

    # Train Phase: Support and Query Set Merge
    sim_data_iterator_train = prepare_sim_dataset_iterator(sim_train_path, "", mid_voc, cat_voc, feature_one_hot_enable=False, l_shots=l_shots)
    # Test Phase: Support Set and Query Set
    sim_data_iterator_support = prepare_sim_dataset_meta_iterator(sim_train_path, None, "", mid_voc, cat_voc, feature_one_hot_enable=False, l_shots=l_shots)
    sim_data_iterator_query = prepare_sim_dataset_meta_iterator(sim_train_path, sim_test_path, "", mid_voc, cat_voc, feature_one_hot_enable=False, l_shots=l_shots)

    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, item_info_path, review_info_path, batch_size, maxlen, shuffle_each_epoch=False)
    print ("DEBUG: Train Dataset Statistics n_uid:%d,n_mid:%d,n_cat_id:%d" % (train_data.n_uid, train_data.n_mid, train_data.n_cat))

    test_data = UserGroupDataIterator(test_file, uid_voc, mid_voc, cat_voc, item_info_path, review_info_path, user_group_batch_size, maxlen, max_batch_size=1)
    n_uid, n_mid, n_cat = train_data.get_n()
    if model_type == 'DNN':
        model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DNN_SIM_NN':
        model = Model_DNN_SIM_NN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DNN_SIM_MAML':
        model = Model_DNN_SIM_MAML(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DNN_SIAMESE':
        model = Model_DNN_SIM_SIAMESE(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DNN_SIM_DEEP_GP':
        model = Model_DNN_SIM_DeepKernelGP(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    else:
        print ("Invalid model_type : %s", model_type)
        return

    # torch training
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.functional.binary_cross_entropy_with_logits
    criterion_mse = nn.MSELoss(size_average = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # Training Iteration
    for epoch_iter in range(total_train_iter):
            loss_sum = 0.0
            loss_sim_sum = 0.0
            matrix_fill_rate_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            train_iter = 0
            # for src, tgt in train_data:
            for src, tgt in train_data:
                train_iter += 1
                #print ("DEBUG: src is" + str(src))
                #print ("DEBUG: tgt is" + str(tgt))
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, maxlen, return_neg=True)
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = torch.LongTensor(uids), torch.LongTensor(mids), torch.LongTensor(cats), torch.LongTensor(mid_his), torch.LongTensor(cat_his), torch.LongTensor(mid_mask), torch.FloatTensor(target), torch.LongTensor(sl), torch.LongTensor(noclk_mids), torch.LongTensor(noclk_cats)

                if model_type == "DNN":
                    data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr]
                    y_logits = model(data)
                    y_true = target[:, 0].unsqueeze(1)
                    loss = criterion(y_logits, y_true)
                    loss2, z_out = None, None
                    matrix_fill_rate = 0.0
                    total_loss = loss

                elif model_type == "DNN_SIM_NN":
                    id_1, id_2, sim_target = prepare_similarity_dataset(sim_data_iterator_train, list(cats.numpy()), if_train=True)
                    id_1, id_2, sim_target = torch.LongTensor(id_1), torch.LongTensor(id_2), torch.FloatTensor(sim_target)
                    no_empty_id_1 = []
                    for x1 in id_1:
                        if x1 != -1:
                            no_empty_id_1.append(x1)
                    matrix_fill_rate = len(no_empty_id_1)/len(id_1)
                    data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr, id_1, id_2, sim_target]

                    y_logits, z_out = model(data)
                    y_true = target[:, 0].unsqueeze(1)
                    loss = criterion(y_logits, y_true)
                    loss2 = criterion_mse(z_out, sim_target.unsqueeze(1))
                    total_loss = loss + w2 * loss2

                elif model_type == "DNN_SIAMESE":
                    id_1, id_2, sim_target = prepare_similarity_dataset(sim_data_iterator_train, list(cats.numpy()), if_train=True)
                    id_1, id_2, sim_target = torch.LongTensor(id_1), torch.LongTensor(id_2), torch.FloatTensor(sim_target)
                    no_empty_id_1 = []
                    for x1 in id_1:
                        if x1 != -1:
                            no_empty_id_1.append(x1)
                    matrix_fill_rate = len(no_empty_id_1)/len(id_1)
                    data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr, id_1, id_2, sim_target]

                    y_logits, z_out = model(data)
                    y_true = target[:, 0].unsqueeze(1)
                    loss = criterion(y_logits, y_true)
                    loss2 = criterion_mse(z_out, sim_target.unsqueeze(1))
                    total_loss = loss + w2 * loss2

                elif model_type == "DNN_SIM_MAML":
                    ## DEBUG: check list(cats)
                    # print ("DEBUG: DNN_SIM_MAML list cats %s" % str(list(cats)))
                    id_1, id_2, sim_target = prepare_similarity_dataset(sim_data_iterator_train, list(cats.numpy()), if_train=True)
                    sim_dataset = [(i1, i2, score) for (i1, i2, score) in zip(id_1, id_2, sim_target)]
                    #print ("DEBUG: sim_dataset is:%s, dataset:%s" % (str(list(cats)), str(sim_dataset)))
                    no_empty_id_1 = []
                    for x1 in id_1:
                        if x1 != -1:
                            no_empty_id_1.append(x1)
                    matrix_fill_rate = len(no_empty_id_1)/len(id_1)
                    recommd_data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr]

                    # sim prediction model
                    sim_pred_model = Model_SIM_PRED_DNN(EMBEDDING_DIM)

                    # print ('DEBUG: Starting Meta Training of Similarity dataset...')
                    maml_tasks = MAML(model, sim_pred_model, sim_dataset, recommd_data, inner_lr=0.001, meta_lr=0.001, inner_steps=1,
                                      task_example_size=l_shots, tasks_per_meta_batch=5)
                    ## update model similarity prediction task weight
                    maml_tasks.main_loop(num_iterations=num_iterations)

                    # forward pass
                    id_1, id_2, sim_target = [w[0] for w in sim_dataset], [w[1] for w in sim_dataset], [w[2] for w in sim_dataset]
                    # item_id_2 = [w[1] for w in sim_dataset]
                    # sim_target = [w[2] for w in sim_dataset]
                    id_1, id_2, sim_target = torch.LongTensor(id_1), torch.LongTensor(id_2), torch.FloatTensor(sim_target)
                    data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr, id_1, id_2, sim_target]
                    y_logits, sim_input = model(data)
                    z_out = sim_pred_model(sim_input)

                    y_true = target[:, 0].unsqueeze(1)
                    loss = criterion(y_logits, y_true)
                    loss2 = criterion_mse(z_out, sim_target.unsqueeze(1))
                    total_loss = loss + w2 * loss2

                elif model_type == "DNN_SIM_DEEP_GP":
                    ## id_1:shape [B*B], id_2: [B*B]
                    id_1, id_2, sim_target = prepare_similarity_dataset(sim_data_iterator_train, list(cats.numpy()), if_train=True, fill_empty=True)
                    no_empty_id_1 = []
                    for x1 in id_1:
                        if x1 != -1:
                            no_empty_id_1.append(x1)
                    matrix_fill_rate = len(no_empty_id_1)/len(id_1)
                    sim_dataset = [(id1, id2, score) for (id1, id2, score) in zip(id_1, id_2, sim_target)]
                    recommd_data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr]

                    # check sim_dataset size
                    if len(sim_dataset) < batch_size * batch_size:
                        print ("DEBUG: Epoch: %d ----> iter: %d ---> Sim Dataset Return Size Not Filled %s" % (epoch_iter, train_iter, str(sim_dataset)))
                        continue
                    # update deep kernel gp_model: net and gp layer
                    dummy_inputs = torch.zeros([l_shots, EMBEDDING_DIM])
                    dummy_labels = torch.zeros([l_shots])
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    gp_model = ExactGPModel(dummy_inputs, dummy_labels, likelihood)
                    deep_gp_tasks = DeepKernelGP(model, gp_model, sim_dataset, recommd_data, recomend_lr=0.001, gp_lr=0.001, inner_steps=5, task_example_size=l_shots)
                    deep_gp_tasks.main_loop(num_gp_iterations=num_iterations)

                    # update similarity tasks
                    task_dataset = deep_gp_tasks.sample_data(sim_dataset)
                    x1 = [w[0] for w in task_dataset]
                    x2 = [w[1] for w in task_dataset]
                    sim_target = [w[2] for w in task_dataset]
                    x1, x2, sim_target = torch.LongTensor(x1), torch.LongTensor(x2), torch.FloatTensor(sim_target)
                    data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr, x1, x2, sim_target]
                    y_logits, sim_input = model(data)
                    # print ("DEBUG: y_logits is %s" % str(y_logits.size()))
                    # print ("DEBUG: sim_input is %s" % str(sim_input.size()))
                    # print ("DEBUG: sim_target is %s" % str(sim_target.size()))
                    gp_model.set_train_data(inputs=sim_input, targets=sim_target)
                    predictions = gp_model(sim_input)

                    if diversify_method == "MMR_UCI":
                        confidence_region = predictions.confidence_region()
                        covariance_matrix = predictions.lazy_covariance_matrix
                        variance_vector = covariance_matrix.diag()
                        z_upper = predictions.mean + alpha * torch.sqrt(variance_vector)

                    y_true = target[:, 0].unsqueeze(1)
                    loss = criterion(y_logits, y_true)
                    loss2 = criterion_mse(predictions.mean, sim_target.unsqueeze(1))
                    # print ("DEBUG: DNN_SIM_DEEP_GP loss2 mse is %f" % loss2)
                    total_loss = loss + loss2

                else:
                    print ("DEBUG: Model Type Not Supported %s" % model_type)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                ## backward of update Total Loss
                total_loss.backward()

                # loss.backward()
                optimizer.step()

                loss_sum += loss
                loss_sim_sum += (loss2 if loss2 is not None else 0.0)
                matrix_fill_rate_sum += matrix_fill_rate
                sys.stdout.flush()
                if (train_iter % log_train_iter) == 0:
                    print('Epoch: %d ----> iter: %d ----> train_loss1: %.4f ---- train loss_sim 1: %.4f ---- sim_matrix_fill_rate_sum: %.4f' % \
                                          (epoch_iter, train_iter, loss_sum / log_train_iter, loss_sim_sum / log_train_iter, matrix_fill_rate_sum / log_train_iter))
                    loss_sum = 0.0
                    loss_sim_sum = 0.0
                    matrix_fill_rate_sum = 0.0
            lr *= 0.5

    ## Test Stage Setting
    rerank_method = None
    if diversify_method == "MMR":
        rerank_method = similarity_utils.MMR
    elif diversify_method == "DPP":
        rerank_method = similarity_utils.DPP
    elif diversify_method == "MMR_UCI":
        rerank_method = similarity_utils.MMR_UCI
    else:
        rerank_method = similarity_utils.MMR

    loss_test = 0.0
    loss_mse_unseen_list = []

    iter_cnt = 0
    K = [1, 5, 10, 20]
    recall_k_batch = []
    ilad_k_batch = []
    cat_div_k_batch = []

    print("----------- Start Testing Stage-----------")
    print("Rerank Method:%s" % str(rerank_method))
    print("Rerank diversity theta:%f" % diversity_theta)
    print("Rerank diversity K:%s" % str(K))

    test_iter = 0
    for src, tgt in test_data:
        test_iter += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt,
                                                                                                        maxlen,
                                                                                                        return_neg=True)
        item_size = len(mids)
        # prediction
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = torch.LongTensor(
            uids), torch.LongTensor(mids), torch.LongTensor(cats), torch.LongTensor(mid_his), torch.LongTensor(
            cat_his), torch.LongTensor(mid_mask), torch.FloatTensor(target), torch.LongTensor(sl), torch.LongTensor(
            noclk_mids), torch.LongTensor(noclk_cats)

        # Meta Test Evaluation MSE
        query_id_unseen_1, query_id_unseen_2, query_sim_target_unseen = prepare_similarity_dataset(
            sim_data_iterator_query, list(cats.numpy()), fill_empty=False, if_train=True)
        # print ("DEBUG: Test phase load query1 %d,query2 %d,query_sim_target size %d" % (
        # len(query_id_unseen_1), len(query_id_unseen_2), len(query_sim_target_unseen)))
        query_id_unseen_1, query_id_unseen_2, query_sim_target_unseen = torch.LongTensor(
            query_id_unseen_1), torch.LongTensor(query_id_unseen_2), torch.FloatTensor(query_sim_target_unseen)
        query_unseen_data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr, query_id_unseen_1,
                             query_id_unseen_2, query_sim_target_unseen]

        # Meta-Test Query Dataset for Full Similarity Matrix
        query_id_1, query_id_2, query_sim_target = prepare_similarity_dataset(sim_data_iterator_query,
                                                                              list(cats.numpy()), if_train=False)
        query_id_1, query_id_2, query_sim_target = torch.LongTensor(query_id_1), torch.LongTensor(
            query_id_2), torch.FloatTensor(query_sim_target)
        query_data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr, query_id_1, query_id_2,
                      query_sim_target]

        if model_type == "DNN":
            data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr]
            with torch.no_grad():
                y_logits = model(data)
                y_true = target[:, 0].unsqueeze(1)
                loss = criterion(y_logits, y_true)
                y_test = y_logits.numpy()
                S_pred = process_similarity_dataset(item_size, torch_array_2_list(mids), torch_array_2_list(cats))
                loss2, loss_mse_unseen, z_out, z_out_unseen = 0.0, 0.0, 0.0, 0.0
        elif (model_type == "DNN_SIM_NN" or model_type == "DNN_SIAMESE"):
            ## rerank user dataiterator train as rerank
            item_id_1, item_id_2, sim_target = prepare_similarity_dataset(sim_data_iterator_query, torch_array_2_list(cats), if_train=False)
            item_id_1, item_id_2, sim_target = torch.LongTensor(item_id_1), torch.LongTensor(item_id_2), torch.FloatTensor(sim_target)
            data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr, item_id_1, item_id_2, sim_target]

            with torch.no_grad():
                y_logits, z_out = model(data)
                y_true = target[:, 0].unsqueeze(1)
                loss = criterion(y_logits, y_true)
                y_test = y_logits.numpy()
                S_pred = process_similarity_from_prediction(item_size, torch_array_2_list(mids), torch_array_2_list(cats), torch_array_2_list(item_id_1), torch_array_2_list(item_id_2), torch_array_2_list(z_out))

                # temp
                y_logits_unseen, z_out_unseen = model(query_unseen_data)
                # print ("DEBUG: z_out_unseen %s, query_sim_target_unseen %s" % (str(z_out_unseen), str(query_sim_target_unseen.unsqueeze(1))))
                loss_mse_unseen = criterion_mse(z_out_unseen, query_sim_target_unseen.unsqueeze(1))
                # print ("DEBUG: loss_mse_unseen is %f" % loss_mse_unseen)
                sim_score_pred_dict, sim_score_true_dict = assembly_similarity_prediction(query_id_unseen_1, query_id_unseen_2, z_out_unseen, sim_target)

        elif model_type == "DNN_SIM_MAML":
            with torch.no_grad():
                # print ("DEBUG: train DNN_SIM_MAML item_id_1 %s" % str(item_id_1))
                # print ("DEBUG: train DNN_SIM_MAML item_id_2 %s" % str(item_id_2))
                y_logits, sim_input = model(query_data)
                z_out = sim_pred_model(sim_input)

                y_true = target[:, 0].unsqueeze(1)
                loss = criterion(y_logits, y_true)
                y_test = y_logits.numpy()
                S_pred = process_similarity_from_prediction(item_size, torch_array_2_list(mids), torch_array_2_list(cats), torch_array_2_list(query_id_1), torch_array_2_list(query_id_2), torch_array_2_list(z_out))
                # Meta Test on un
                # meta-test for query only unseen similarity pattern
                y_logits_unseen, z_out_unseen = model(query_unseen_data)
                # temp
                # print ("DEBUG: z_out_unseen %s, query_sim_target_unseen %s" % (str(z_out_unseen), str(query_sim_target_unseen.unsqueeze(1))))
                loss_mse_unseen = criterion_mse(z_out_unseen, query_sim_target_unseen.unsqueeze(1))
                sim_score_pred_dict, sim_score_true_dict = assembly_similarity_prediction(query_id_unseen_1, query_id_unseen_2, z_out_unseen, sim_target)

        elif model_type == "DNN_SIM_DEEP_GP":
            # Meta Test Support Set
            support_id_1, support_id_2, support_sim_target = prepare_similarity_dataset(sim_data_iterator_support, torch_array_2_list(cats), if_train=True)
            support_id_1, support_id_2, support_sim_target = torch.LongTensor(support_id_1), torch.LongTensor(support_id_2), torch.FloatTensor(support_sim_target)
            support_data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr, support_id_1, support_id_2, support_sim_target]
            # Meta Test Query Set
            query_id_1, query_id_2, query_sim_target = prepare_similarity_dataset(sim_data_iterator_query, torch_array_2_list(cats), if_train=False)
            query_id_1, query_id_2, query_sim_target = torch.LongTensor(query_id_1), torch.LongTensor(query_id_2), torch.FloatTensor(query_sim_target)
            query_data = [uids, mids, mid_his, cats, cat_his, mid_mask, sl, target, lr, query_id_1, query_id_2, query_sim_target]

            with torch.no_grad():
                # meta-train
                y_logits_support, support_sim_input = model(support_data)
                ## DEBUG:
                gp_model.train()
                gp_model.set_train_data(inputs=support_sim_input, targets=support_sim_target, strict=False)

                # meta-test for query full similarity pattern
                y_logits_query, query_sim_input = model(query_data)
                gp_model.eval()
                #  = gp_model(query_sim_input.detach())
                ll = likelihood(gp_model(query_sim_input.detach()))
                mean = ll.mean
                z_out = mean

                if diversify_method == "MMR_UCI":
                    confidence_region = ll.confidence_region()
                    covariance_matrix = ll.lazy_covariance_matrix
                    variance_vector = covariance_matrix.diag()
                    z_out = mean + 1.95 * torch.sqrt(variance_vector)
                    # print ("DEBUG: DNN_SIM_DEEP_GP y_logits_support test z_upper_bound %s, z_original %s, variance %s" % (str(z_out), str(mean), str(variance_vector)))

                # z_out = gp_model(sim_input.detach())
                y_true = target[:, 0].unsqueeze(1)
                loss = criterion(y_logits_query, y_true)
                y_test = y_logits_query.numpy()
                S_pred = process_similarity_from_prediction(item_size, torch_array_2_list(mids), torch_array_2_list(cats), torch_array_2_list(query_id_1), torch_array_2_list(query_id_2), torch_array_2_list(z_out))

                # meta-test for query only unseen similarity pattern
                y_logits_unseen, query_sim_input_unseen = model(query_unseen_data)
                gp_model.eval()
                z_out_unseen = likelihood(gp_model(query_sim_input_unseen.detach())).mean

                # temp
                # print ("DEBUG: z_out_unseen mean %s, query_sim_target_unseen %s" % (str(z_out_unseen), str(query_sim_target_unseen.unsqueeze(1))))
                loss_mse_unseen = criterion_mse(z_out_unseen, query_sim_target_unseen.unsqueeze(1))
                # print ("DEBUG: loss_mse_unseen is %f" % loss_mse_unseen)
                sim_score_pred_dict, sim_score_true_dict = assembly_similarity_prediction(query_id_unseen_1, query_id_unseen_2, z_out_unseen, sim_target)
        else:
            print ("DEBUG: Not Supported...")

        loss_test += loss
        # loss_mse_unseen_sum += loss_mse_unseen
        if math.isnan(loss_mse_unseen):
            loss_mse_unseen = 0.0
        loss_mse_unseen_list.append(loss_mse_unseen)
        iter_cnt += len(list(uids))

        original_id_list = [w[0] for w in sorted([(i, score) for i, score in enumerate(list(y_test))], key=lambda x:x[1], reverse=True)]
        id_list = rerank_method(y_test, S_pred, theta=diversity_theta)
        clk_item_id = get_clk_item_id(item_size, mids, target)
        index_2_cat_map = {}
        for i, cat_id in enumerate(list(cats.numpy())):
            index_2_cat_map[i] = cat_id

        if test_iter % log_test_iter == 0 and False:
            print ('Test Stage loss %f--------' % loss)
            print ("DEBUG: iter %d loss is %s" % (test_iter, str(loss)))
            print ("DEBUG: iter %d mids is %s" % (test_iter, str(mids)))
            print ("DEBUG: iter %d cats is %s" % (test_iter, str(cats)))
            print ("DEBUG: iter %d target is %s" % (test_iter, str(target)))
            print ("DEBUG: iter %d y_test is %s" % (test_iter, str(y_test)))
            print ("DEBUG: iter %d z_out is %s" % (test_iter, str(z_out)))
            print ("DEBUG: iter %d z_out_unseen is %s" % (test_iter, str(z_out_unseen)))
            print ("DEBUG: iter %d loss_mse_unseen is %s" % (test_iter, str(loss_mse_unseen)))
            print ("DEBUG: iter %d Reranked index_2_cat_map is %s" % (test_iter, str(index_2_cat_map)))
            print ("DEBUG: iter %d item_size is %s" % (test_iter, str(item_size)))
            print ("DEBUG: iter %d S_pred is %s" % (test_iter, str(S_pred)))
            print ("DEBUG: iter %d Reranked original_id_list is %s" % (test_iter, str(original_id_list)))
            print ("DEBUG: iter %d Reranked new_id_list is %s" % (test_iter, str(id_list)))
            print ("DEBUG: iter %d Reranked clk_item_id is %s" % (test_iter, str(clk_item_id)))

        # S_ground_truth = process_similarity_ground_truth_dataset(item_size, mids, cats, sim_data_iterator_query)
        S_ground_truth = process_similarity_ground_truth_dataset(item_size, torch_array_2_list(mids), torch_array_2_list(cats), sim_data_iterator_query)
        recall_k_list, ilad_k_list, cat_div_k_list = eval_diversity(id_list, clk_item_id, K, S_ground_truth, index_2_cat_map)
        recall_k_batch.append(recall_k_list)
        ilad_k_batch.append(ilad_k_list)
        cat_div_k_batch.append(cat_div_k_list)
        # LOG
        if test_iter % log_test_iter == 0:
            print ("DEBUG: -------- Evaluation at test iter %d -------------" % test_iter)
            print ("DEBUG: -------- DEBUG: recall_k \t ilad_k \t cat_div_k -------------")
            recall_k_mean = np.mean(np.array(recall_k_batch), axis=0)
            ilad_k_mean = np.mean(np.array(ilad_k_batch), axis=0)
            cat_div_k_mean = np.mean(np.array(cat_div_k_batch), axis=0)
            for i, k in enumerate(K):
                print ("DEBUG: Recall@%d %f \t %f \t %f " % (k, recall_k_mean[i], ilad_k_mean[i], cat_div_k_mean[i]))
            test_mse = np.mean(loss_mse_unseen_list) if len(loss_mse_unseen_list) > 0 else 0.0
            print ("DEBUG: -------- DEBUG: iter %d Loss MSE on unseen similarity query set %f -------------" % (test_iter, test_mse))

            print ("DEBUG: -------- DEBUG: iter %d S_ground_truth is %s -------------" % (test_iter, str(S_ground_truth)))
