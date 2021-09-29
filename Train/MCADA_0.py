import os
import torch
from MC import data_path, networks, get_data, train
import csv
import numpy as np

torch.cuda.set_device(1)  # set GPU .


feas_dim = 2048
num_classes = 12
batch_size = 128
iterations = 10

root_path = data_path.ImageCLEF_root_path


def pre_train_fine_tune(ds, dt, bs_ds=batch_size, bs_dt=None, decay_lr=True, normalization=True,
                        feas_type='Resnet50', n_dtl=0.03, feas_dim=feas_dim, out_dim=num_classes):

    if not bs_dt:
        bs_dt = bs_ds

    root_path = data_path.ImageCLEF_root_path
    domain_name = '{}_{}'.format(ds[:-4], dt[:-4])
    print(domain_name)
    result_path = r'./Result'
    os.makedirs(result_path, exist_ok=True)

    # train_epochs：all train_epochs，tgt_discriminator, src_discriminator, src_classifier,  save_epochs
    train_epochs = [500, 5, 5, 5]
    # domain labels: ds, dt, ds_f, dt_f
    domain_labels = [1.0, 0.0, 1.0, 0.1]
    # loss_c_ds, loss_d, loss_d_f, loss_c_ds_f, distance, entropy_loss
    loss_weights = [1, 0.5, 0.5, 1, 0.1, 0.1]
    update_gradients = [2, 2, 3]

    for n_clusters_dt_pre_train in [3]:
        feas_ds, labels_ds, feas_dt, labels_dt = get_data.get_ds_dt_feas_labels_dtl_by_components(
            root_path, ds, dt, feas_type, int(n_clusters_dt_pre_train), n_dtl
        )
        dataloader_ds = get_data.get_dataloader_by_feas_labels(
            feas_ds, labels_ds, bs_ds, normalization=normalization, fea_type=feas_type)
        dataloader_dt = get_data.get_dataloader_by_feas_labels(
            feas_dt, labels_dt, bs_dt, normalization=normalization, fea_type=feas_type)
        dataloader = [dataloader_ds, dataloader_dt]

        # pre-train
        for _ in range(iterations):
            discriminator = networks.Discriminator(in_dim=feas_dim).cuda()
            classifier = networks.Classifier(in_dim=feas_dim, out_dim=out_dim).cuda()
            acc_tgt_best, best_classifier, best_discriminator = train.train_adv(
                discriminator, classifier, dataloader, train_epochs, domain_labels, update_gradients,
                      loss_weights, decay_lr=decay_lr, lr=1e-4)

            with open(r'{}\{}.csv'.format(result_path, domain_name), 'a+',
                      newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow([float(acc_tgt_best)])


if __name__ == '__main__':
    import time
    start = time.time()

    pre_train_fine_tune(data_path.domain_c, data_path.domain_ci)
    print(time.time() - start)
    # pre_train_fine_tune(data_path.domain_c, data_path.domain_cp)
    # #
    # pre_train_fine_tune(data_path.domain_i, data_path.domain_ic)
    # pre_train_fine_tune(data_path.domain_i, data_path.domain_ip)
    # #
    # pre_train_fine_tune(data_path.domain_p, data_path.domain_pc)
    # pre_train_fine_tune(data_path.domain_p, data_path.domain_pi)