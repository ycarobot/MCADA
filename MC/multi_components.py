import torch
import numpy as np
from MC import get_data

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def guassian_kernel(ds, dt, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # rbf function
    n_samples = len(ds) + len(dt)
    total = torch.cat([ds, dt], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
    total1 = total.unsqueeze(1).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # multi kernnels
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def get_distance_mmd_rbf_accelerate(ds, dt, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    bs = min(ds.shape[0], dt.shape[0], 200)
    if len(ds) > bs:
        index = np.random.choice(np.arange(len(ds)), bs, replace=False)
        ds = ds[index]
    if len(dt) > bs:
        index = np.random.choice(np.arange(len(dt)), bs, replace=False)
        dt = dt[index]
    kernels = guassian_kernel(ds[:bs], dt[:bs],
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(bs):
        s1, s2 = i, (i+1)%bs
        t1, t2 = s1+bs, s2+bs
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / (float(bs) + 1e-9)


def get_distance_mmd_rbf_no_accelerate(ds, dt, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    bs = min(len(ds), len(dt), 200)
    if len(ds) > bs:
        index = np.random.choice(np.arange(len(ds)), bs, replace=False)
        ds = ds[index]
    if len(dt) > bs:
        index = np.random.choice(np.arange(len(dt)), bs, replace=False)
        dt = dt[index]
    kernels = guassian_kernel(ds, dt, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:bs, :bs]
    YY = kernels[bs:, bs:]
    XY = kernels[:bs, bs:]
    YX = kernels[bs:, :bs]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss.item()



def get_MMD_martix(feas_ds_list, feas_dt_list):
    distance_matrix = np.ones((len(feas_ds_list), len(feas_dt_list)))
    for i in range(len(feas_ds_list)):
        for j in range(len(feas_dt_list)):
            mmd_distance = get_distance_mmd_rbf_accelerate(
                FloatTensor(feas_ds_list[i]), FloatTensor(feas_dt_list[j]))
            distance_matrix[i, j] = mmd_distance

    return distance_matrix


def get_pairs_by_greedy(d_matrix):
    n_Ds, n_Dt = d_matrix.shape
    index_pairs_Dt = np.argmin(d_matrix, axis=0)
    pairs = [(src, tgt) for tgt, src in enumerate(index_pairs_Dt)]
    # return (ds, dt) (ds, dt)
    return pairs


def concatenate_other_pairs_data(feas_list, labels_list, current_pair_i, n_other_pairs_data,
                                 pairs_distance_max_min='min'):
    # concatenate data, Ds or Dt
    # select samples from other pairs
    feas_src, labels_src = feas_list[current_pair_i], labels_list[current_pair_i]
    # get pairs index
    other_pairs_index = [i for i in range(len(feas_list)) if i != current_pair_i]

    # calculate distance
    distance_list = [get_distance_mmd_rbf_accelerate(
        FloatTensor(feas_src), FloatTensor(feas_list[i])) for i in other_pairs_index]
    pairs_index = 0

    if pairs_distance_max_min == 'min':
        pairs_index = other_pairs_index[int(np.argmin(distance_list))]
    elif pairs_distance_max_min == 'max':
        pairs_index = other_pairs_index[int(np.argmax(distance_list))]

    # random choice
    extra_index = np.random.choice(len(feas_list[pairs_index]), int(n_other_pairs_data * len(feas_list[pairs_index])))
    feas_extra, labels_extra = get_data.separate_data(feas_list[pairs_index], labels_list[pairs_index], extra_index)
    feas_src, labels_src = get_data.concatenate_data(feas_src, labels_src, feas_extra, labels_extra)

    return feas_src, labels_src


def get_concatenate_dataloader(feas_ds_list, labels_ds_list, feas_dt_list, labels_dt_list, current_pair_i,
                               concatenate_mode, pairs_distance_max_min, n_other_pairs_data, bs_ds=32, bs_dt=32,
                               normalization=True, feas_type='Resnet50'):

    if concatenate_mode == 'Ds':
        # concatenate only for Ds
        feas_ds, labels_ds = concatenate_other_pairs_data(
            feas_ds_list, labels_ds_list, current_pair_i=current_pair_i[0],
            n_other_pairs_data=n_other_pairs_data, pairs_distance_max_min=pairs_distance_max_min)
        dataloader_src = get_data.get_dataloader_by_feas_labels(
            feas_ds, labels_ds, drop_last=False, bs=bs_ds, normalization=normalization, fea_type=feas_type)
        dataloader_tgt = get_data.get_dataloader_by_feas_labels(
            feas_dt_list[current_pair_i[1]], labels_dt_list[current_pair_i[1]], drop_last=False, bs=bs_dt,
        normalization=normalization, fea_type=feas_type)

    elif concatenate_mode == 'Dt':
        # concatenate only for Dt
        dataloader_src = get_data.get_dataloader_by_feas_labels(
            feas_ds_list[current_pair_i[0]], labels_ds_list[current_pair_i[0]], drop_last=False, bs=bs_ds,
            normalization=normalization, fea_type=feas_type)
        feas_tgt, labels_tgt = concatenate_other_pairs_data(
            feas_dt_list, labels_dt_list, current_pair_i=current_pair_i[1],
            n_other_pairs_data=n_other_pairs_data, pairs_distance_max_min=pairs_distance_max_min)
        dataloader_tgt = get_data.get_dataloader_by_feas_labels(
            feas_tgt, labels_tgt, drop_last=False, bs=bs_dt, normalization=normalization, fea_type=feas_type)

    else:
        # concatenate for Ds and Dt
        feas_ds, labels_ds = concatenate_other_pairs_data(
            feas_ds_list, labels_ds_list, current_pair_i=current_pair_i[0],
            n_other_pairs_data=n_other_pairs_data, pairs_distance_max_min=pairs_distance_max_min)
        dataloader_src = get_data.get_dataloader_by_feas_labels(
            feas_ds, labels_ds, drop_last=False, bs=bs_ds, normalization=normalization, fea_type=feas_type)
        feas_tgt, labels_tgt = concatenate_other_pairs_data(
            feas_dt_list, labels_dt_list, current_pair_i=current_pair_i[1],
            n_other_pairs_data=n_other_pairs_data, pairs_distance_max_min=pairs_distance_max_min)
        dataloader_tgt = get_data.get_dataloader_by_feas_labels(
            feas_tgt, labels_tgt, drop_last=False, bs=bs_dt, normalization=normalization, fea_type=feas_type)

    return dataloader_src, dataloader_tgt


