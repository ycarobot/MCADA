import os
import torch
from MC import data_path, networks, get_data, train, multi_components, MCADA
import csv
import numpy as np


torch.cuda.set_device(0)  # set GPU .


fea_dim = 2048
num_classes = 12
batch_size = 256
iterations = 5

root_path = data_path.ImageCLEF_root_path




def pre_train_fine_tune(ds, dt, iterations=3, batch_size=batch_size, n_dtl=0.03, fea_type='Resnet50'):

    domain_name = '{}_{}'.format(ds[:-4], dt[:-4])
    result_path = r'../Result'
    os.makedirs(result_path, exist_ok=True)

    # pre-train
    train_params = {'train_epochs': [400, 2, 2, 2], 'domain_labels': [1.0, 0.0, 1.0, 0.1], 'batch_size': batch_size,
                    'loss_weights': [1, 0.5, 0.5, 1, 0.1, 0.1], 'update_gradients': [2, 2, 3], 'n_dtl': n_dtl, 'lr': 1e-4,
                    'normalization': True, 'fea_type': fea_type, 'fea_dim': fea_dim, 'num_classes': num_classes,
                    'decay_lr': True}


    acc_tgt_best, best_classifier, best_discriminator, best_data_list = MCADA.Stage1(
        ds, dt, root_path, train_params=train_params)

    # fine-tune
    train_params = {'train_epochs': [100, 2, 2, 2], 'domain_labels': [1.0, 0.0, 1.0, 0.1], 'batch_size': batch_size,
                    'loss_weights': [1, 0.5, 0.5, 1, 0.1, 0.1], 'update_gradients': [2, 2, 3], 'n_dtl': n_dtl,
                    'lr': 1e-4, 'iterations': iterations,  'normalization': True, 'fea_type': fea_type,
                    'fea_dim': fea_dim, 'num_classes': num_classes,
                    'decay_lr': True}
    MCADA.Stage2(
        best_data_list, best_classifier, best_discriminator, result_path, domain_name,
        concatenate_other_data=True, train_params=train_params)


if __name__ == '__main__':

    pre_train_fine_tune(data_path.domain_c, data_path.domain_ci)
    pre_train_fine_tune(data_path.domain_c, data_path.domain_cp)

    pre_train_fine_tune(data_path.domain_i, data_path.domain_ic)
    pre_train_fine_tune(data_path.domain_i, data_path.domain_ip)

    pre_train_fine_tune(data_path.domain_p, data_path.domain_pc)
    pre_train_fine_tune(data_path.domain_p, data_path.domain_pi)