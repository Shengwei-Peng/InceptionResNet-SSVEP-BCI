###############################################################################
# 相關套件導入
###############################################################################
from torchsignal.datasets.hsssvep import HSSSVEP
from torchsignal.datasets.multiplesubjects import MultipleSubjects

import numpy as np
import torch
from torch import nn

from torchsummary import summary 
from sklearn.metrics import accuracy_score
import copy
import time
import torch.nn.functional as F
###############################################################################
# 參數設置
###############################################################################
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
  "root": "torchsignal\datasets",
  
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
###############################################################################
# 載入資料
###############################################################################
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
###############################################################################
# 載入模型
###############################################################################
class Model(nn.Module):
    def __init__(self, num_channel=10, num_classes=4, signal_length=1000, filters_n1=4, kernel_window_ssvep=59, kernel_window=19, conv_3_dilation=4):
        super().__init__()
        
        filters = [filters_n1, filters_n1 * 2]
        
        self.conv_1 = conv_block(in_ch=1, out_ch=filters[0], kernel_size=(1, kernel_window_ssvep) ,padding=(0,(kernel_window_ssvep-1)//2))
        self.conv_2 = conv_block(in_ch=filters[0], out_ch=filters[0], kernel_size=(num_channel, 1))
        self.conv_3_1 = conv_block(in_ch=filters[0], out_ch=filters[1], kernel_size=(1, kernel_window), padding=(0,(kernel_window-1)*(conv_3_dilation-2)), dilation=(1,conv_3_dilation))
        self.conv_3_2 = conv_block(in_ch=filters[1], out_ch=filters[1], kernel_size=(1, kernel_window), padding=(0,(kernel_window-1)*(conv_3_dilation-2)), dilation=(1,conv_3_dilation))

        self.pool = nn.MaxPool2d(kernel_size=(1,2))
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(signal_length*filters[1]//2,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,num_classes)
        
        self.Residual_1 = Residual_Block(in_ch=filters[0], out_ch=filters[0], kernel_1=kernel_window, kernel_2=(kernel_window+kernel_window_ssvep)//2, kernel_3=kernel_window_ssvep)
        self.Residual_2 = Residual_Block(in_ch=filters[1], out_ch=filters[1], kernel_1=kernel_window, kernel_2=(kernel_window+kernel_window_ssvep)//2, kernel_3=kernel_window_ssvep)
        
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        
        x = self.conv_1(x)
        x = self.Residual_1(x)
        x = self.dropout(x)
        
        x = self.conv_2(x)
        x = self.dropout(x)
        
        x = self.conv_3_1(x)
        x = self.Residual_2(x)
        x = self.pool(x)
        
        x = self.conv_3_2(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
###############################################################################
# 自訂義層
###############################################################################
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=(0,0), dilation=(1,1), w_in=None):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )
        if w_in is not None:
            self.w_out = int( ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1) / 1) + 1 )
            
    def forward(self, x):
        return self.conv(x)
    
class Residual_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_1, kernel_2, kernel_3): 
        super(Residual_Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.Inception = Inception_Block(in_ch=in_ch, out_ch=out_ch, kernel_1= kernel_1, kernel_2= kernel_2, kernel_3 =  kernel_3)
        
    def forward(self, x):
        residual = x
        out = self.Inception(x)
        out += residual
        out = self.relu(out)

        return out

class Inception_Block(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_1, kernel_2, kernel_3):
        super( Inception_Block,self).__init__()
        out = int(out_ch//4)
        self.kernel_2 = kernel_2
        self.branch1_1 = nn.Conv2d(in_ch,out,kernel_size=(1,kernel_1),padding=(0,(kernel_1-1)//2))
        
        self.branch2_1 = nn.Conv2d(in_ch,in_ch,kernel_size=(1,kernel_1),padding=(0,(kernel_1-1)//2))
        self.branch2_2 = nn.Conv2d(in_ch,out,kernel_size=(1,kernel_3),padding=(0,(kernel_3-1)//2))
 
        self.branch3_1 = nn.Conv2d(in_ch,in_ch,kernel_size=(1,kernel_1),padding=(0,(kernel_1-1)//2))
        self.branch3_2 = nn.Conv2d(in_ch,out,kernel_size=(1,kernel_2),padding=(0,(kernel_2-1)//2))
        self.branch3_3 = nn.Conv2d(out,out,kernel_size=(1,kernel_2),padding=(0,(kernel_2-1)//2))
 
        self.branch_pool = nn.Conv2d(in_ch,out,kernel_size=(1,kernel_1),padding=(0,(kernel_1-1)//2))
 
    def forward(self,x):
        branch1_1 = self.branch1_1(x)
 
        branch2_1 = self.branch2_1(x)
        branch2_2 = self.branch2_2(branch2_1)
 
        branch3_1 = self.branch3_1(x)
        branch3_2 = self.branch3_2(branch3_1)
        branch3_3 = self.branch3_3(branch3_2)
 
        branch_pool = F.avg_pool2d(x,kernel_size=(1,self.kernel_2),stride=1,padding=(0,(self.kernel_2-1)//2))
        branch_pool = self.branch_pool(branch_pool)
 
        outputs = [branch_pool, branch1_1, branch2_2, branch3_3]
        return torch.cat(outputs,dim=1)
###############################################################################
# 模型測試
###############################################################################
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
###############################################################################
# 訓練與測試副程式
###############################################################################
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
    return epoch_loss, np.round(epoch_acc.item(), 3), classification_f1

def get_preds(outputs, k=1):
    _, preds = outputs.topk(k, 1, True, True)
    preds = preds.t()
    return preds[0]

def get_parameters():
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
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
###############################################################################
# 模型訓練
###############################################################################
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
    
    epochs=config['epochs'] if 'epochs' in config else 50
    patience=config['patience'] if 'patience' in config else 20
    early_stopping=config['early_stopping'] if 'early_stopping' in config else 40
    
    SEED = config['seed']
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    optimizer = torch.optim.Adam(get_parameters(), lr=config["learning_rate"], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)
    fit(model, dataloaders_dict, num_epochs=epochs, early_stopping=early_stopping, topk_accuracy=1, save_model=False)
###############################################################################
# 訓練結果
###############################################################################
    test_loss, test_acc, test_metric = validate(model, test_loader, 1)
    print('Testset (S{}) -> loss:{:.5f}, acc:{:.5f}'.format(subject_id, test_loss, test_acc))
    acc.append(test_acc)

print('準確率:', np.mean(acc))