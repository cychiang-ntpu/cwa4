import logging
import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from cwa4.dataset import Dataset4
from cwa4.encoder import Classifier1


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler('train.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

logger = setup_logger()
device = torch.device("cuda:0")

for target_width in (1, 90, 180, 365, 730):
    # training loop
    dataset = Dataset4(
        gnss_path="hualian_daily_gnss_dXdYdU.pkl", 
        statistics_path="hualian_daily_statistics.pkl", 
        target_path="hualian_target_cnt.pkl", 
        input_width=730, 
        target_width=target_width,
        subset="trn"
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = Classifier1(dataset.input_dim, h_dim=128)
    model.to(device)  # move model to device
    
    optim = Adam(model.parameters(),lr=0.0001)

    loss_fn = BCELoss()
    logger.info("start traning")
    model.train()
    for epoch in range(5):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            loss = loss_fn(model(x), y)
            loss.backward()
            logger.info(f"epoch: {epoch}, loss: {loss.item()}")
    torch.save(model.state_dict(), f"state_dict_{target_width}.pt")
    # 驗證
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    tp, fp, tn, fn = 0, 0, 0, 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x).round()
        # 計算 accuracy 
        tp += torch.sum( (y == 1.0) & (y_hat == 1.0) ).item()
        fp += torch.sum( (y == 0.0) & (y_hat == 1.0) ).item()
        tn += torch.sum( (y == 0.0) & (y_hat == 0.0) ).item()
        fn += torch.sum( (y == 1.0) & (y_hat == 0.0) ).item()
    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn) 

    logger.info(f"metrics of the training set. target_width: {target_width}")
    logger.info(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, N={tp+fp+tn+fn}")
    logger.info(f"accuracy: {accuracy*100.0:.4f}%, precision: {precision*100.0:.4f}%, recall: {recall*100.0:.4f}")
    #  Test set
    dataset = Dataset4(
        gnss_path="hualian_daily_gnss_dXdYdU.pkl", 
        statistics_path="hualian_daily_statistics.pkl", 
        target_path="hualian_target_cnt.pkl", 
        input_width=730, 
        target_width=target_width,
        subset="tst"
    )
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
            
    tp, fp, tn, fn = 0, 0, 0, 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x).round()
        # 計算 accuracy 
        tp += torch.sum( (y == 1.0) & (y_hat == 1.0) ).item()
        fp += torch.sum( (y == 0.0) & (y_hat == 1.0) ).item()
        tn += torch.sum( (y == 0.0) & (y_hat == 0.0) ).item()
        fn += torch.sum( (y == 1.0) & (y_hat == 0.0) ).item()
    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn) 

    logger.info(f"metrics of the test set. target_width: {target_width}")
    logger.info(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, N={tp+fp+tn+fn}")
    logger.info(f"accuracy: {accuracy*100.0:.4f}%, precision: {precision*100.0:.4f}%, recall: {recall*100.0:.4f}")

