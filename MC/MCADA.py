
import torch
from MC import data_path, networks, get_data, train, multi_components
import csv
import numpy as np
import copy



def Stage1(ds, dt, root_path, train_params=None):
    train_epochs = train_params['train_epochs']
    domain_labels = train_params['domain_labels']
    loss_weights = train_params['loss_weights']
    update_gradients = train_params['update_gradients']
    acc_tgt_best = best_classifier = best_discriminator = best_data_list = 0

    for n_clusters_dt in [14]:
        feas_ds, labels_ds, feas_dt, labels_dt = get_data.get_ds_dt_feas_labels_dtl_by_components(
            root_path, ds, dt, train_params['fea_type'], int(n_clusters_dt), train_params['n_dtl']
        )
        dataloader_ds = get_data.get_dataloader_by_feas_labels(
            feas_ds, labels_ds, train_params['batch_size'], normalization=train_params['normalization'], fea_type=train_params['fea_type'])
        dataloader_dt = get_data.get_dataloader_by_feas_labels(
            feas_dt, labels_dt, train_params['batch_size'], normalization=train_params['normalization'], fea_type=train_params['fea_type'])
        dataloader = [dataloader_ds, dataloader_dt]

        discriminator = networks.Discriminator(in_dim=train_params['fea_dim']).cuda()
        classifier = networks.Classifier(in_dim=train_params['fea_dim'], out_dim=train_params['num_classes']).cuda()
        acc_tgt, classifier, discriminator = train.train_adv(
            discriminator, classifier, dataloader, train_epochs, domain_labels, update_gradients,
            loss_weights, decay_lr=train_params['decay_lr'], lr=train_params['lr'])
        if acc_tgt > acc_tgt_best:
            acc_tgt_best = acc_tgt
            best_classifier = classifier
            best_discriminator = discriminator
            best_data_list = [feas_ds, labels_ds, feas_dt, labels_dt]
    return acc_tgt_best, best_classifier, best_discriminator, best_data_list


def Stage2(
        best_data_list, best_classifier, best_discriminator, result_path, domain_name, concatenate_other_data=True,
        train_params=None):
    train_epochs = train_params['train_epochs']
    domain_labels = train_params['domain_labels']
    loss_weights = train_params['loss_weights']
    update_gradients = train_params['update_gradients']
    for n_c_dt in [2, 3, 4, 5, 6, 7, 8]:
        for n_c_ds in [2, 3, 4, 5, 6, 7, 8]:
            classifier0 = networks.Classifier(
                in_dim=train_params['fea_dim'], out_dim=train_params['num_classes'])
            classifier0.load_state_dict(best_classifier.state_dict())
            feas_ds_list, labels_ds_list = get_data.clustering(
                classifier0(torch.FloatTensor(best_data_list[0])).detach().numpy(), best_data_list[1], n_c_ds, best_data_list[0])
            feas_dt_list, labels_dt_list = get_data.clustering(
                classifier0(torch.FloatTensor(best_data_list[2])).detach().numpy(), best_data_list[3], n_c_dt, best_data_list[2])
            d_matrix = multi_components.get_MMD_martix(feas_ds_list=feas_ds_list, feas_dt_list=feas_dt_list)
            clusters_pairs = multi_components.get_pairs_by_greedy(d_matrix)
            for _ in range(train_params['iterations']):
                dt_nums_pair_acc = dt_nums_pair_all = 0
                for clusters_pair in clusters_pairs:
                    # network reload
                    discriminator_pair = networks.Discriminator(in_dim=train_params['fea_dim']).cuda()
                    classifier_pair = networks.Classifier(in_dim=train_params['fea_dim'], out_dim=train_params['num_classes']).cuda()
                    discriminator_pair.load_state_dict(best_discriminator.state_dict())
                    classifier_pair.load_state_dict(best_classifier.state_dict())

                    dataloader_ds0 = get_data.get_dataloader_by_feas_labels(
                        feas=feas_ds_list[clusters_pair[0]], labels=labels_ds_list[clusters_pair[0]],
                        bs=train_params['batch_size'], normalization=['normalization'], fea_type=train_params['fea_type']
                    )
                    dataloader_dt0 = get_data.get_dataloader_by_feas_labels(
                        feas=feas_dt_list[clusters_pair[1]], labels=labels_dt_list[clusters_pair[1]],
                        bs=train_params['batch_size'], normalization=['normalization'], fea_type=train_params['fea_type']
                    )
                    # struct dataloader
                    if concatenate_other_data:
                        dataloader_ds, dataloader_dt = multi_components.get_concatenate_dataloader(
                            feas_ds_list=feas_ds_list, labels_ds_list=labels_ds_list, feas_dt_list=feas_dt_list,
                            labels_dt_list=labels_dt_list, current_pair_i=clusters_pair, concatenate_mode='Ds_Dt',
                            pairs_distance_max_min='max', n_other_pairs_data=0.5, normalization=['normalization'],
                            feas_type=train_params['fea_type'])
                    else:
                        dataloader_ds = dataloader_ds0
                        dataloader_dt = dataloader_dt0
                    dataloader = [dataloader_ds, dataloader_dt]
                    dataloader0 = [dataloader_ds0, dataloader_dt0]
                    print('Length. Ds0:{}||Ds:{} Dt0:{}||Dt:{}'.format(
                        len(dataloader_ds0.dataset), len(dataloader_ds.dataset), len(dataloader_dt0.dataset), len(dataloader_dt.dataset)
                    ))

                    acc_tgt_best, _, _ = train.train_adv(
                        discriminator_pair, classifier_pair, dataloader, train_epochs, domain_labels, update_gradients,
                        loss_weights, decay_lr=train_params['decay_lr'], lr=train_params['lr'], dataloader0=dataloader0)

                    dt_nums_pair_acc += float(acc_tgt_best * len(dataloader_dt0.dataset))
                    dt_nums_pair_all += len(dataloader_dt0.dataset)

                with open('{}/{}.csv'.format(result_path, domain_name), 'a+', newline='') as f:
                    f_csv = csv.writer(f)
                    # if _ == 0:
                    #     information = '[&domain name:{} &n_Dtl:{} &clusters :{}||{}][ &decay_lr:{}]'. \
                    #         format(domain_name, n_dtl, n_clusters_ds, n_clusters_dt, decay_lr)
                    #     f_csv.writerow([information])
                    f_csv.writerow([float(dt_nums_pair_acc / (dt_nums_pair_all + 1e-6))])
