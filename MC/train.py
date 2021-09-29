import torch.optim as optim
from torch.autograd.variable import *
from MC.Optimizer_functions import *
from MC import entropy_functions, multi_components
import copy

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor


def evaluate(classifier, dataloader_src, dataloader_tgt):

    classifier.eval()
    acc_src = acc_tgt = 0
    for (imgs_tgt, labels_tgt) in dataloader_tgt:
        feature_tgt = Variable(imgs_tgt.type(FloatTensor).reshape(imgs_tgt.shape[0], -1))
        labels_tgt = Variable(labels_tgt.type(LongTensor))

        predict_prob_tgt = classifier(feature_tgt)
        pred_tgt = predict_prob_tgt.data.max(1)[1]
        acc_tgt += pred_tgt.eq(labels_tgt.data).cpu().sum()

    for (imgs_src, labels_src) in dataloader_src:
        feature_src = Variable(imgs_src.type(FloatTensor)).reshape(imgs_src.shape[0], -1)
        labels_src = Variable(labels_src.type(LongTensor))

        predict_prob_src = classifier(feature_src)
        pred_src = predict_prob_src.data.max(1)[1]
        acc_src += pred_src.eq(labels_src.data).cpu().sum()
    acc_src = int(acc_src) / len(dataloader_src.dataset)
    acc_tgt = int(acc_tgt) / len(dataloader_tgt.dataset)
    print("[Src Accuracy = {:2%}, Tgt Accuracy = {:2%}]".format(acc_src, acc_tgt))
    classifier.train()
    return acc_src, acc_tgt


def get_adv_feas(classifier, discriminator, data_dict, train_epochs, update_grads, domain_labels):
    criterion_c = torch.nn.CrossEntropyLoss()
    criterion_d = torch.nn.BCEWithLogitsLoss()

    feas_ds, feas_dt, labels_ds = data_dict['feas_ds'], data_dict['feas_dt'], data_dict['labels_ds']
    feas_ds_f = Variable(feas_ds.type(FloatTensor), requires_grad=True).reshape(feas_ds.shape[0], -1)
    feas_ds_f0 = feas_ds_f.detach()
    feas_dt_f = Variable(feas_dt.type(FloatTensor), requires_grad=True).reshape(feas_dt.shape[0], -1)
    feas_dt_f0 = feas_dt_f.detach()

    labels_ds = Variable(labels_ds.type(LongTensor))

    for i_t_d in range(train_epochs[0]):
        discriminator.zero_grad()
        classifier.zero_grad()
        domain_labels_dt_f = Variable(FloatTensor(feas_dt_f.size(0), 1).fill_(domain_labels[3]))
        loss_d = criterion_d(discriminator(feas_dt_f), domain_labels_dt_f) - 0.1 * torch.sum(
            (feas_dt_f - feas_dt_f0) * (feas_dt_f - feas_dt_f0))
        feas_dt_f.retain_grad()
        loss_d.backward()
        feas_dt_f = feas_dt_f + update_grads[0] * feas_dt_f.grad
        feas_dt_f = Variable(feas_dt_f, requires_grad=True)

    for i_s_d in range(train_epochs[1]):
        discriminator.zero_grad()
        classifier.zero_grad()
        domain_labels_ds_f = Variable(FloatTensor(feas_ds_f.size(0), 1).fill_(domain_labels[2]))
        loss_d = criterion_d(discriminator(feas_ds_f), domain_labels_ds_f) - 0.1 * torch.sum(
            (feas_ds_f - feas_ds_f0) * (feas_ds_f - feas_ds_f0))
        feas_ds_f.retain_grad()
        loss_d.backward()
        feas_ds_f = feas_ds_f + update_grads[1] * feas_ds_f.grad
        feas_ds_f = Variable(feas_ds_f, requires_grad=True)

    for i_s_c in range(train_epochs[2]):
        discriminator.zero_grad()
        classifier.zero_grad()
        loss_c = criterion_c(classifier(feas_ds_f), labels_ds) - 0.1 * torch.sum(
            (feas_ds_f - feas_ds_f0) * (feas_ds_f - feas_ds_f0))
        feas_ds_f.retain_grad()
        loss_c.backward()
        feas_ds_f = feas_ds_f + update_grads[2] * feas_ds_f.grad
        feas_ds_f = Variable(feas_ds_f, requires_grad=True)

    return feas_ds_f, feas_dt_f


def train_adv(discriminator, classifier, dataloader, train_epochs, domain_labels, update_gradients,
              loss_weights, decay_lr=True, lr=1e-4, dataloader0=None):
    if dataloader0 is None:
        dataloader0 = dataloader
    criterion_d = torch.nn.BCEWithLogitsLoss()
    criterion_c = torch.nn.CrossEntropyLoss()
    if decay_lr:
        scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=3000)
        optimizer_c = OptimWithSheduler(optim.Adam(classifier.parameters(), weight_decay=5e-4, lr=lr),
                                        scheduler)
        optimizer_d = OptimWithSheduler(optim.Adam(discriminator.parameters(), weight_decay=5e-4, lr=lr),
                                        scheduler)
    else:
        optimizer_c = optim.Adam(classifier.parameters(), lr=lr, weight_decay=5e-4)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=5e-4)
    acc_tgt_best = best_classifier = best_discriminator = iter_ds = iter_dt = 0

    for epoch in range(train_epochs[0]):
        classifier.train()
        discriminator.train()
        if epoch % len(dataloader[0]) == 0:
            iter_ds = iter(dataloader[0])
        if epoch % len(dataloader[1]) == 0:
            iter_dt = iter(dataloader[1])

        feas_ds, labels_ds = iter_ds.__next__()
        feas_dt, labels_dt = iter_dt.__next__()
        if feas_ds.shape[0] == 1 or feas_dt.shape[0] == 1:
            continue
        data_dict = {'feas_ds': feas_ds, 'labels_ds': labels_ds, 'feas_dt': feas_dt}

        feas_ds = Variable(feas_ds.type(FloatTensor)).reshape(feas_ds.shape[0], -1)
        labels_ds = Variable(labels_ds.type(LongTensor))
        feas_dt = Variable(feas_dt.type(FloatTensor)).reshape(feas_dt.shape[0], -1)

        feas_ds_f, feas_dt_f = get_adv_feas(
            classifier=classifier, discriminator=discriminator, data_dict=data_dict, train_epochs=train_epochs[1:],
            update_grads=update_gradients, domain_labels=domain_labels
        )

        # classifier output
        output_c_ds = classifier(feas_ds)
        output_c_dt = classifier(feas_dt)
        output_c_ds_f = classifier(feas_ds_f)
        output_c_dt_f = classifier(feas_dt_f)

        # discriminator output
        output_d_ds = discriminator(feas_ds)
        output_d_dt = discriminator(feas_dt)
        output_d_dt_f = discriminator(feas_dt_f)
        output_d_ds_f = discriminator(feas_ds_f)

        # domain labels
        domain_labels_ds = Variable(FloatTensor(feas_ds.size(0), 1).fill_(domain_labels[0]))
        domain_labels_dt = Variable(FloatTensor(feas_dt.size(0), 1).fill_(domain_labels[1]))
        domain_labels_ds_f = Variable(FloatTensor(feas_ds.size(0), 1).fill_(domain_labels[2]))
        domain_labels_dt_f = Variable(FloatTensor(feas_dt.size(0), 1).fill_(domain_labels[3]))

        # Loss
        loss_d_f = (criterion_d(output_d_ds_f.detach(), domain_labels_ds_f) +
                     criterion_d(output_d_dt_f.detach(), domain_labels_dt_f))
        loss_d = criterion_d(output_d_ds, domain_labels_ds) + \
                  criterion_d(output_d_dt, domain_labels_dt)
        loss_c_ds = criterion_c(output_c_ds, labels_ds)
        entropy_loss = entropy_functions.entropy_loss(output_c_dt)
        distance = torch.sum((output_c_dt_f - output_c_dt) *
                         (output_c_dt_f - output_c_dt))
        loss_c_ds_f = criterion_c(output_c_ds_f, labels_ds)

        with OptimizerManager([optimizer_c, optimizer_d]):
            loss = loss_weights[0] * loss_c_ds + loss_weights[1] * loss_d + loss_weights[2] * loss_d_f + \
               loss_weights[3] * loss_c_ds_f + loss_weights[4] * distance + loss_weights[5] * entropy_loss
            loss.backward()

        if epoch % 30 == 0:
            acc_src, acc_tgt = evaluate(classifier, dataloader0[0], dataloader0[1])
            if acc_tgt > acc_tgt_best:
                acc_tgt_best = acc_tgt
                best_classifier, best_discriminator = copy.deepcopy(classifier), copy.deepcopy(discriminator)


    return acc_tgt_best, best_classifier, best_discriminator



