import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from framework.utilities import create_folder
from framework.models_pytorch import move_data_to_gpu
import framework.config as config
from sklearn import metrics


def define_system_name(alpha=None, basic_name='system', batch_size=None):
    suffix = ''
    if alpha:
        suffix = suffix.join([str(each) for each in alpha]).replace('.', '')

    sys_name = basic_name
    sys_suffix = '_b' + str(batch_size)

    sys_suffix = sys_suffix + '_cuda' + str(config.cuda_seed) if config.cuda_seed is not None else sys_suffix
    system_name = sys_name + sys_suffix if sys_suffix is not None else sys_name

    return suffix, system_name



 
def forward_asc_aec(model, generate_func, cuda):
    outputs = []
    outputs_event = []

    targets = []
    targets_event = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_y, batch_y_event) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        # print(batch_x.size())

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x)  # torch.Size([16, 10])
            batch_rate, batch_output_event = all_output[0], all_output[1]

            batch_output_event = F.sigmoid(batch_output_event)

            outputs.append(batch_rate.data.cpu().numpy())
            outputs_event.append(batch_output_event.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dict['outputs_event'] = outputs_event

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event
    return dict


def evaluate_asc_aec(model, generator, data_type, cuda):

    # Generate function
    generate_func = generator.generate_validate(data_type=data_type)

    # Forward
    dict = forward_asc_aec(model=model, generate_func=generate_func, cuda=cuda)

    # rate loss
    targets = dict['output']
    predictions = dict['target']
    # rate_mse_loss = metrics.mean_squared_error(targets, predictions)
    rate_mse_loss = metrics.mean_squared_error(targets, predictions, squared=False)
    # rmse

    # aec
    outputs_event = dict['outputs_event']  # (audios_num, classes_num)
    targets_event = dict['targets_event']  # (audios_num, classes_num)
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc = sum(aucs) / len(aucs)

    return rate_mse_loss, final_auc


def training_testing(generator, model, cuda, models_dir, epochs, batch_size, lr_init = 1e-3):
    create_folder(models_dir)

    # Optimizer 
    optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    sample_num = len(generator.train_audio_ids)
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')

    val_loss_rate = []
    val_auc_event = []

    # Train on mini batches
    for iteration, all_data in enumerate(generator.generate_train()):

        (batch_x, batch_y_cpu, batch_y_event_cpu) = all_data
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y_cpu, cuda)
        batch_y_event = move_data_to_gpu(batch_y_event_cpu, cuda)

        model.train()
        optimizer.zero_grad()

        x_rate_linear, x_event_linear = model(batch_x)

        x_event_sigmoid = F.sigmoid(x_event_linear)
        loss_event = bce_loss(x_event_sigmoid, batch_y_event)

        loss_rate = mse_loss(x_rate_linear, batch_y)

        loss_common = loss_rate + loss_event

        loss_common.backward()
        optimizer.step()

        print('epoch: ', '%.4f' % (iteration/one_epoch), 'loss: %.5f' % float(loss_common),
              'l_rate: %.5f' % float(loss_rate),
              'l_event: %.5f' % float(loss_event ),)

        # Stop learning
        if iteration > (epochs * one_epoch):
            print('Training is done!!!')
            break

    # testing
    va_rate_mse_loss, va_event_auc = evaluate_asc_aec(model=model,
                                                      generator=generator,
                                                      data_type='validate',
                                                      cuda=cuda)

    val_loss_rate.append(va_rate_mse_loss)
    val_auc_event.append(va_event_auc)

    print('E: ', '%.4f' % (iteration/one_epoch), ' test_rate_loss: %.3f' % va_rate_mse_loss,
          ' test_event_auc: %.3f' % va_event_auc)

    # Save model
    save_out_dict = {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
    save_out_path = os.path.join(models_dir, 'model' + config.endswith)
    torch.save(save_out_dict, save_out_path)