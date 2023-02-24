import torch.utils.data
from data.vid_train_dataset import WaterDropDataset as Vid_Train_WaterDropDataset
from data.img_train_dataset import RainDropDataset as Img_Train_RainDropDataset
from data.vid_test_dataset_syn import WaterDropDataset as Vid_Test_WaterDropDataset_Syn
from data.vid_test_dataset_real import WaterDropDataset as Vid_Test_WaterDropDataset_Real

def CreateDataLoader(opt):
    if opt.isTrain:
        vid_train_dataset = Vid_Train_WaterDropDataset(opt)
        img_train_dataset = Img_Train_RainDropDataset(opt)
        
        vid_train_loader = torch.utils.data.DataLoader(vid_train_dataset, batch_size=opt.batchSize, shuffle=opt.shuffle,
                                                       num_workers=int(opt.nThreads), drop_last=True)

        img_train_loader = torch.utils.data.DataLoader(img_train_dataset, batch_size=opt.batchSize*opt.n_frames, shuffle=opt.shuffle,
                                                       num_workers=int(opt.nThreads), drop_last=True)
        
        return vid_train_loader, img_train_loader
    else:
        vid_test_dataset_syn = Vid_Test_WaterDropDataset_Syn(opt)
        vid_test_dataset_real = Vid_Test_WaterDropDataset_Real(opt)

        vid_test_loader_syn = torch.utils.data.DataLoader(vid_test_dataset_syn, batch_size=1, shuffle=False,
                                                          num_workers=int(opt.nThreads), drop_last=True)
        vid_test_loader_real = torch.utils.data.DataLoader(vid_test_dataset_real, batch_size=1, shuffle=False,
                                                              num_workers=int(opt.nThreads), drop_last=True)

        return vid_test_loader_syn, vid_test_loader_real
