import argparse
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.metrics import f1_score
from glob import glob
from models import *
from utils import *
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import os

def get_predict_npy(cfg):
    dataset = pd.read_csv(data_dir+'sample_submission.csv',dtype={'Predicted':str})

    print(cfg)

    test = dataset.copy()
    test.columns = ['Id','Target']
    test['Id'] = data_dir + 'img_data/' + test['Id']
    test['suffix'] = '.png'
    gens = [HumanDataset(test, 'test', cfg, None, get_augumentor(f'TTA{i}')) for i in range(12)]
    dataloaders = [DataLoader(gen, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=8) for gen in gens]


    model,_ = cfg['model'](cfg)

    result = []
    # result2 = []
    for fold in cfg['fold']:
        print(fold,'--------------------')
        if os.path.exists(f"{cfg['name']}fold{fold}.npy"):
            pred = np.load(f"{cfg['name']}fold{fold}.npy")
        else:
            model.load_state_dict(torch.load(f"../weights/{cfg['name']}_fold{fold}.pkl"))
            pred_list = []
            if len(cfg['device']) > 1:
                parallel_model = nn.DataParallel(model, device_ids=cfg['device']).cuda()
                for dataloader in dataloaders:
                    _, pred = predict(parallel_model, dataloader, False)
                    pred_list.append(pred)
            else:
                model.cuda()
                for dataloader in dataloaders:
                    _, pred = predict(model, dataloader, False)
                    pred_list.append(pred)
            pred = np.average(pred_list, axis=0)
            np.save(f"{cfg['name']}fold{fold}", pred)
        result.append(pred)
    result = np.average(result,axis=0)
    np.save(f"{cfg['name']}submit",result)

def main(cfg):
    dataset = pd.read_csv(data_dir + 'sample_submission.csv', dtype={'Predicted': str})
    result = np.load(f"{cfg['name']}submit.npy")

    target = []
    for sample_pred in result:
        pred = []
        for i,score in enumerate(sample_pred):
            if score > cfg['thres'][i]:
                pred.append(str(i))
        if len(pred) == 0:
            pred.append(str(sample_pred.argmax()))
        target.append(' '.join(pred))

    dataset['Predicted'] = target
    dataset.to_csv('submit.csv',index=False)





if __name__ == '__main__':
    labels_dict = [
        40958.0,
        3072.0,
        10871.0,
        3329.0,
        5130.0,  # 4
        5938.0,
        3725.0,
        9405.0,
        217.0,
        197.0,  # 9
        182.0,
        2194.0,
        2233.0,
        1458.0,
        2692.0,  # 14
        63.0,
        1290.0,
        446.0,
        1893.0,
        3672.0,  # 19
        438.0,
        13809.0,
        2729.0,
        10345.0,
        428.0,  # 24
        37366.0,
        706.0,
        127.0
    ]
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--fold', type=str, required=True)
    # parser.add_argument('--name', type=str, required=True)
    # parser.add_argument('--device', type=str, required=True)
    from config import *

    # args = parser.parse_args()
    cfg = {}
    # cfg['fold'] = [int(fold) for fold in args.fold]
    # cfg['device'] = [i for i in range(len(args.device))]
    cfg['model'] = resnet18_model
    cfg['lr'] = 0.0001
    cfg['backbone_lr'] = cfg['lr']
    cfg['bs'] = 128
    cfg['channel'] = 4
    cfg['name'] = 'res34_shisu'
    cfg['aug'] = True
    cfg['thres'] = [0.3]*28
    cfg['save_oof'] = True
    cfg['use_external_data'] = True
    cfg['half'] = False
    # get_predict_npy(cfg)
    # get_oof_data(cfg)
    main(cfg)

    # cfg['thres'] = [0.5]*28
    # main(cfg)



