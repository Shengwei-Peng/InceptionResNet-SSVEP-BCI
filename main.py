import copy
import time
import torch
from torchsummary import summary 
import numpy as np
from sklearn.metrics import accuracy_score
from torchsignal.datasets.hsssvep import HSSSVEP
from torchsignal.datasets.multiplesubjects import MultipleSubjects
from models import Model

config = {
  "exp_name": "model",
  "seed": 12,
  "segment_config": {
    "window_len": 4,
    "shift_len": 250,
    "sample_rate": 250,
    "add_segment_axis": True
  },
  "bandpass_config": {
      "sample_rate": 250,
      "lowcut": 1,
      "highcut": 40,
      "order": 6
  },
  "train_subject_ids": {
    "low": 1,
    "high": 35
  },
  "test_subject_ids": {
    "low": 1,
    "high": 35
  },
  "root": "torchsignal/datasets",
  
  "selected_channels": ['PZ', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2', 'PO7', 'PO8'],
  "num_classes": 40,
  "num_channel": 11,
  "batchsize": 64,
  "learning_rate": 0.001,
  "epochs": 100,
  "patience": 5,
  "early_stopping": 100,
  "model": {
    "n1": 4,
    "kernel_window_ssvep": 59,
    "kernel_window": 19,
    "conv_3_dilation": 4,
    "conv_4_dilation": 4
  },
  "gpu": 0,
  "multitask": True,
  "runkfold": 5,
}

device = torch.device("cuda:"+str(config['gpu']) if torch.cuda.is_available() else "cpu")
print('device', device)

subject_ids = list(np.arange(config['train_subject_ids']['low'], config['train_subject_ids']['high']+1, dtype=int))

data = MultipleSubjects(
    dataset=HSSSVEP, 
    root=config['root'], 
    subject_ids=subject_ids, 
    selected_channels=config['selected_channels'],
    segment_config=config['segment_config'],
    bandpass_config=config['bandpass_config'],
    one_hot_labels=True,
)

print("Input data shape:", data.data_by_subjects[1].data.shape)
print("Target shape:", data.data_by_subjects[1].targets.shape)


model = Model(num_channel=config['num_channel'],
    num_classes=config['num_classes'],
    signal_length=config['segment_config']['window_len'] * config['bandpass_config']['sample_rate'],
    filters_n1=config['model']['n1'],
    kernel_window_ssvep=config['model']['kernel_window_ssvep'],
    kernel_window=config['model']['kernel_window'],
    conv_3_dilation=config['model']['conv_3_dilation'],
).to(device)

summary(model, (data.data_by_subjects[1].data.shape[1], data.data_by_subjects[1].data.shape[2]))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Model size:', count_params(model))

del model

def train(model, data_loader, topk_accuracy):
    model.train()
    return _loop(data_loader, train_mode=True, topk_accuracy=topk_accuracy)

def validate(model, data_loader, topk_accuracy):
    model.eval()
    return _loop(data_loader, train_mode=False, topk_accuracy=topk_accuracy)

def _loop(data_loader, train_mode=True, topk_accuracy=1):
    running_loss = 0.0
    num_classes = config['num_classes']
    y_true = []
    y_pred = []
    criterion = nn.CrossEntropyLoss()
    for X, Y in data_loader:
        inputs = X.to(device)
        labels = Y.float().to(device)
        if train_mode:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) * outputs.size(0)
        labels = np.argmax(labels.data.cpu().numpy(), axis=1)
        y_true.extend(labels)
        y_pred.extend(get_preds(outputs, topk_accuracy).cpu().numpy())
        running_loss += loss.item() * num_classes
        if train_mode:
            loss.backward()
            optimizer.step()
    epoch_loss = running_loss / len(y_true)
    epoch_acc = accuracy_score(y_true, y_pred)
    classification_f1 = 0
    return epoch_loss, np.round(epoch_acc, 3), classification_f1

def get_preds(outputs, k=1):
    _, preds = outputs.topk(k, 1, True, True)
    preds = preds.t()
    return preds[0]

def get_parameters():
    params_to_update = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    return params_to_update

def fit(model, dataloaders_dict, num_epochs=10, early_stopping=5, topk_accuracy=1, min_num_epoch=10, save_model=False):
    print("-------")
    print("Starting training, on device:", device)

    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    early_stopping_counter = early_stopping

    best_epoch_info = {
        'model_wts':copy.deepcopy(model.state_dict()),
        'loss':1e10
    }

    for epoch in range(num_epochs):
        time_epoch_start = time.time()
        train_loss, train_acc, train_classification_f1 = train(model, dataloaders_dict['train'], topk_accuracy)
        val_loss, val_acc, val_classification_f1 = validate(model, dataloaders_dict['val'], topk_accuracy)
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(val_acc)
        improvement = False
        if val_loss < best_epoch_info['loss']:
            improvement = True
            best_epoch_info = {
                'model_wts':copy.deepcopy(model.state_dict()),
                'loss':val_loss,
                'epoch':epoch,
                'metrics':{
                    'train_loss':train_loss,
                    'val_loss':val_loss,
                    'train_acc':train_acc,
                    'val_acc':val_acc,
                    'train_classification_f1':train_classification_f1,
                    'val_classification_f1':val_classification_f1
                }
            }

        if early_stopping and epoch > min_num_epoch:
            if improvement:
                early_stopping_counter = early_stopping
            else:
                early_stopping_counter -= 1

            if early_stopping_counter <= 0:
                print("Early Stop")
                break
        if val_loss < 0:
            print('val loss negative')
            break
        print("Epoch {:2} in {:.0f}s || Train loss={:.3f}, acc={:.3f} | Val loss={:.3f}, acc={:.3f} | LR={:.1e} | best={} | improvement={}-{}".format(
            epoch+1,
            time.time() - time_epoch_start,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            optimizer.param_groups[0]['lr'],
            int(best_epoch_info['epoch'])+1,
            improvement,
            early_stopping_counter)
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        else:
            scheduler.step()

acc = []
test_subject_ids = list(np.arange(config['test_subject_ids']['low'], config['test_subject_ids']['high']+1, dtype=int))

for subject_id in test_subject_ids:
    print('Subject', subject_id)

    train_loader, val_loader, test_loader = data.leave_one_subject_out(selected_subject_id=subject_id, dataloader_batchsize=config['batchsize'])
    dataloaders_dict = {
        'train': train_loader,
        'val': val_loader
        }

    model = Model(num_channel=config['num_channel'],
        num_classes=config['num_classes'],
        signal_length=config['segment_config']['window_len'] * config['bandpass_config']['sample_rate'],
        filters_n1=config['model']['n1'],
        kernel_window_ssvep=config['model']['kernel_window_ssvep'],
        kernel_window=config['model']['kernel_window'],
        conv_3_dilation=config['model']['conv_3_dilation'],
    ).to(device)

    epochs = config['epochs'] if 'epochs' in config else 50
    patience = config['patience'] if 'patience' in config else 20
    early_stopping = config['early_stopping'] if 'early_stopping' in config else 40

    SEED = config['seed']
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    optimizer = torch.optim.Adam(get_parameters(), lr=config["learning_rate"], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)
    fit(model, dataloaders_dict, num_epochs=epochs, early_stopping=early_stopping, topk_accuracy=1, save_model=False)

    test_loss, test_acc, test_metric = validate(model, test_loader, 1)
    print(f'Testset (S{subject_id}) -> loss:{test_loss:.5f}, acc:{test_acc:.5f}')
    acc.append(test_acc)

print('The mean accuracy from Leave One Out Cross-Validation:', np.mean(acc))
