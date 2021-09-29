import numpy as np
import os
import torch.utils.data as data
import torch
from sklearn.cluster import SpectralClustering, KMeans
from scipy.io import loadmat
from MC import normalization_data


def separate_data(feas, labels, index, in_index=True):
    if in_index:
        feas = np.asarray([feas[i] for i in index])
        labels = np.asarray([labels[i] for i in index])

    else:
        feas = np.asarray([feas[i] for i in range(len(feas)) if i not in index])
        labels = np.asarray([labels[i] for i in range(len(labels)) if i not in index])
    return feas, labels


def concatenate_data(feas, labels, feas_extra, labels_extra):
    if len(np.asarray(feas).shape) == len(np.asarray(feas_extra).shape):
        feas = np.concatenate((feas, feas_extra), 0)
        labels = np.concatenate((labels, labels_extra), 0)
    return  feas, labels


def list_numpy(feas_list, labels_list):
    # list to numpy
    feas = feas_list[0]
    labels = labels_list[0]
    for i in range(1, len(feas_list)):
        feas = np.concatenate((feas, feas_list[i]), 0)
        labels = np.concatenate((labels, labels_list[i]), 0)
    return feas, labels


def get_feas_labels(root_path, domain, fea_type='Resnet50'):
    # get feas and labels by feas_type
    path = os.path.join(root_path, domain)
    if fea_type == 'Resnet50':
        with open(path, encoding='utf-8') as f:
            imgs_data = np.loadtxt(f, delimiter=",")
            features = imgs_data[:, :-1]
            labels = imgs_data[:, -1]

    elif fea_type == 'MDS':
        domain_data = loadmat(path)
        features = np.asarray(domain_data['fts'])
        labels = np.asarray(domain_data['labels']).squeeze()

    else: # DeCAF6
        domain_data = loadmat(path)
        features = np.asarray(domain_data['feas'])
        labels = np.asarray(domain_data['labels']).squeeze() - 1  # start from 0
    return features, labels


def get_dataloader_by_feas_labels(feas, labels, bs=128, drop_last=False, normalization=True, fea_type='Resnet50'):
    if normalization:
        # data normalization
        dataset = normalization_data.myDataset(feas, labels, fea_type)
    else:
        dataset = data.TensorDataset(torch.tensor(feas), torch.tensor(labels))
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=drop_last,
    )
    return dataloader


def clustering(feas_c_out_put, labels, n_clusters, feas_original, return_labels=False):
    # get components
    # return feas_list and labels_list
    k = KMeans(n_clusters=n_clusters)
    y_pred = k.fit_predict(feas_c_out_put)

    # return list
    feas_list = [[feas_original[i] for i in range(len(feas_original)) if y_pred[i] == n] for n in range(n_clusters)]
    labels_list = [[labels[i] for i in range(len(labels)) if y_pred[i] == n] for n in range(n_clusters)]

    if return_labels:
        return feas_list, labels_list, k.labels_
    else:
        return feas_list, labels_list


def get_ds_dt_feas_labels_dtl_by_components(root_path, ds, dt, fea_type, n_clusters_dt, n_dtl):
    # dtl selected based on components
    feas_ds, labels_ds = get_feas_labels(root_path, ds, fea_type=fea_type)
    feas_dt, labels_dt = get_feas_labels(root_path, dt, fea_type=fea_type)

    # Clustering Dt
    feas_dt_list, labels_dt_list = clustering(feas_dt, labels_dt, n_clusters_dt, feas_original=feas_dt)

    # get dtl in every component
    feas_dt = labels_dt = 0
    for i in range(n_clusters_dt): # get dtl in every component
        # print('[Length feas_tgt_list{}: {}]'.format(i, len((feas_tgt_list[i]))))
        dtl_index = np.random.choice(len(feas_dt_list[i]), int(n_dtl * len(feas_dt_list[i])), replace=False)
        feas_dtl, labels_dtl = separate_data(feas_dt_list[i], labels_dt_list[i], dtl_index)
        feas_dtu, labels_dtu = separate_data(feas_dt_list[i], labels_dt_list[i], dtl_index, in_index=False)
        # dtl to ds
        feas_ds, labels_ds = concatenate_data(feas_ds, labels_ds, feas_dtl, labels_dtl)

        if i == 0:
            feas_dt, labels_dt = feas_dtu, labels_dtu
        else:
            feas_dt, labels_dt = concatenate_data(feas_dt, labels_dt, feas_dtu, labels_dtu)

    return feas_ds, labels_ds, feas_dt, labels_dt


def get_ds_dtl_dtu(feas_ds, labels_ds, feas_dt, labels_dt, n_dtl):
    extra_index = np.random.choice(len(feas_dt), int(n_dtl * len(feas_dt)), replace=False)

    fea_tgt_label = np.asarray([feas_dt[i] for i in extra_index])
    labels_tgt_label = np.asarray([labels_dt[i] for i in extra_index])

    feas_dt = np.asarray([feas_dt[i] for i in range(len(feas_dt)) if i not in extra_index])
    labels_dt = np.asarray([labels_dt[i] for i in range(len(labels_dt)) if i not in extra_index])

    feas_ds = np.concatenate((feas_ds, fea_tgt_label), 0)
    labels_ds = np.concatenate((labels_ds, labels_tgt_label), 0)

    return feas_ds, labels_ds, feas_dt, labels_dt


def get_sd_td_with_labels_dataloader(root_path, domain_src, domain_tgt, n_Dtl, fea_type, batch_size=100):
    feas_src, labels_src = get_feas_labels(root_path, domain_src, fea_type=fea_type)
    feas_tgt, labels_tgt = get_feas_labels(root_path, domain_tgt, fea_type=fea_type)

    feas_src, labels_src, feas_tgt, labels_tgt = get_ds_dtl_dtu(feas_src, labels_src, feas_tgt, labels_tgt, n_Dtl)

    fea_type = data.TensorDataset(torch.tensor(feas_src), torch.tensor(labels_src))
    dataloader_src = data.DataLoader(
        dataset=fea_type,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    fea_type = data.TensorDataset(torch.tensor(feas_tgt), torch.tensor(labels_tgt))
    dataloader_tgt = data.DataLoader(
        dataset=fea_type,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    return dataloader_src, dataloader_tgt
