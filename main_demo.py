import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from utils import *

import random
import numpy as np
import os, datetime
import sys
import logging
from tqdm import tqdm

from networks import EDSR, SANet, MedU, MGDUN
from data.dataset_IXI import DatasetIXI, RandomGenerator
from networks.metrics import *

import argparse

def build_args():
    # parent_parser = ArgumentParser(add_help=False)

    # parser = MInterface.add_model_specific_args(parent_parser)
    # parser = Trainer.add_argparse_args(parser)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Basic Training Control
    parser.add_argument('--exp_name', type=str, default='MGDUN_IXI',              # TODO exp_name
                            help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')                   # TODO GPU numbers

    parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='satup_data', type=str)
    parser.add_argument('--data_dir', default='dataset', type=str)
    parser.add_argument('--model_name', default='EDSR', type=str)
    parser.add_argument('--loss', default='l1', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    
    # Data Hyperparameters
    parser.add_argument(
            "--sample_rate", default=1.0, type=float,
        )

    # Model Hyperparameters
    parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
    parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
    
    parser.add_argument('--sigma', type=int, default=3,
                    help='gaussian kernel')

    parser.add_argument('--model', default='EDSR',
                    help='model name')
    parser.add_argument('--stage', type=int, default=4,
                    help='the number of stage')

    parser.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parser.add_argument('--pre_train', type=str, default='',
                        help='pre-trained model directory')
    parser.add_argument('--extend', type=str, default='.',
                        help='pre-trained model directory')
    parser.add_argument('--n_resblocks', type=int, default=16,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--dilation', action='store_true',
                        help='use dilated convolution')
    # parser.add_argument('--precision', type=str, default='single',
    #                     choices=('single', 'half'),
    #                     help='FP precision for test (single | half)')

    # Other
    parser.add_argument('--color_range', default=255, type=int)
    parser.add_argument('--aug_prob', default=0.5, type=float)

    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=200)

    args = parser.parse_args()

    return args

def print_options(opt, LOGGING):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            # default = parser.get_default(k)
            # if v != default:
            #     comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '------------------- End -----------------\n'
        LOGGING.info(message)

def main(args):
    # if not args.deterministic:
    #     cudnn.benchmark = True
    #     cudnn.deterministic = False
    # else:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    now_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    snapshot_path = os.path.join(os.getcwd(), args.exp_name, now_time+"_stage_"+str(args.stage)+ "_sigma_"+str(args.sigma)) 

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # model
    model = MGDUN(1,
                args.n_feats, 
                args.n_resblocks,  
                args.color_range, 
                args.n_colors, 
                args.res_scale,
                scale = int(args.scale),
                iter_num=args.stage).cuda()
    

    logging.basicConfig(filename=snapshot_path + "/log_run_" + args.exp_name + ".txt", level=logging.INFO,   # TODO logging file path
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    logging.info(f'{args.model_name}: Number of params: {sum([p.data.nelement() for p in model.parameters()])}')
    logging.info(f'Number of {args.model_name} learnable params: {get_number_of_learnable_parameters(model)}')

    print_options(args, logging)
    logging.info(f'snapshot:{snapshot_path}' )

    # data_load 

    batch_size = args.batch_size
    trainset = DatasetIXI(scale=args.scale, mode='train', sigma=args.sigma,
                        transform=transforms.Compose(
                                   [RandomGenerator()]))
    valset = DatasetIXI(scale=args.scale, mode='val', sigma=args.sigma,
                        transform=transforms.Compose(
                                   [RandomGenerator()]))
    testset = DatasetIXI(scale=args.scale, mode='test', sigma=args.sigma,
                        transform=transforms.Compose(
                                   [RandomGenerator()]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_load = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    val_load = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_load = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # trainset = DatasetIXI(data_dir='**/train', args=args)
    # valset = DatasetIXI(data_dir='**/val', args=args, validtion_flag=True)
    # testset = DatasetIXI(data_dir='**/test', args=args, validtion_flag=True)
    # train_load = DataLoader(
    #                 dataset=trainset,
    #                 batch_size=args.batch_size,
    #                 num_workers=1,
    #                 pin_memory=False,
    #                 drop_last=True,
    #                 # sampler=sampler,
    #             )
    # val_load = DataLoader(
    #                 dataset=valset,
    #                 batch_size=args.batch_size,
    #                 num_workers=1,
    #                 pin_memory=False,
    #                 drop_last=False,
    #                 # sampler=sampler,
    #             )
    # test_load = DataLoader(
    #                 dataset=testset,
    #                 batch_size=args.batch_size,
    #                 num_workers=1,
    #                 pin_memory=False,
    #                 drop_last=False,
    #                 # sampler=sampler,
    #             )
    model.train()

    if args.loss == 'mse':
        loss_function = F.mse_loss
    elif args.loss == 'l1':
        loss_function = F.l1_loss
    else:
        raise ValueError("Invalid Loss Type!")

    base_lr = args.lr
    # optimizer = torch.optim.RMSprop(
    #         model.parameters(), lr=base_lr, weight_decay=args.weight_decay,
    #     )
    optimizer = torch.optim.Adam(
            model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optim, args.lr_step_size, args.lr_gamma
    # )

    # TODO tensorboard files path
    writer = SummaryWriter(snapshot_path + '/log_run_' + args.exp_name)

    epoch_begin = 0
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_load)  # max_epoch = max_iterations // len(trainloader) + 1

    # # TODO 若不存在the_best_model,则加载预训练模型数据, 若存在，则加载model中的数据继续训练
    # if not os.path.exists(os.path.join(snapshot_path, 'best_model.pth')):
    #     if args.n_gpu > 1:
    #         model = nn.DataParallel(model)
    # else:
    #     if args.n_gpu > 1:
    #         model = nn.DataParallel(model)
    #     state_dict = torch.load(os.path.join(snapshot_path, 'best_model.pth'))
    #     model.load_state_dict(state_dict['model_state_dict'])
    #     if optim is not None:
    #         optim.load_state_dict(state_dict['optimizer'])
    #     epoch_begin = state_dict['epoch'] + 1
    #     iter_num = epoch_begin * len(train_load)


    eval_score_higher_is_better = True
    if eval_score_higher_is_better:
        best_eval_score = float('-inf')
    else:
        best_eval_score = float('+inf')
    best_eval_psnr = 0
    best_eval_ssim = 0
    best_eval_cc = 0
    best_eval_mse = 0
    # iterator = tqdm(range(epoch_begin, max_epoch), ncols=70)

    # train
    # for epoch_num in iterator:
    #     iterator.set_description('Epoch %d' % epoch_num)
    for epoch_num in range(epoch_begin, max_epoch):
        logging.info('the epoch number: %d ' % (epoch_num))
        with tqdm(total=len(train_load), desc="train phase") as pbar:
            for i_batch, sampled_batch in enumerate(train_load):
                D_T2, T2, PD = sampled_batch
                D_T2, T2, PD = D_T2.cuda(), T2.cuda(), PD.cuda()
                U_T2 = model(D_T2, PD)
                loss = loss_function(U_T2, T2)

                # pbar.set_postfix_str({'iteration: {}  loss: {}'.format(iter_num, loss)})
                # pbar.update()

                pbar.set_postfix(iteration=iter_num, loss=loss.item())
                pbar.update()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                # for param_group in optimizer.param_groups:
                    # param_group['lr'] = lr_

                # logs = {"loss": loss.detach()}

                iter_num = iter_num + 1
                # writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/l1_loss', loss, iter_num)
                # logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

        # TODO test & save the best model (epoch)
        test_interval = 1
        # if epoch_num > int(max_epoch * 0.8) and (epoch_num + 1) % test_interval == 0:
        
        # if epoch_num > int(max_epoch * 0.5) and (epoch_num + 1) % test_interval == 0:
        if (epoch_num + 1) % test_interval == 0:
            with tqdm(total=len(test_load), desc="test phase") as pbar_test:
                model.eval()
                with torch.no_grad():
                    # snapshot = os.path.join(snapshot_path, 'best_model.pth')
                    # if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
                    # net.load_state_dict(torch.load(snapshot))
                    snapshot_name = snapshot_path.split('/')[-1]
                    # TODO predictions path
                    if args.is_savenii:
                        # args.test_save_dir = os.path.join(dataset_config[dataset_name]['save_base_path'], args.exp_name, "predictions_run")
                        # test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
                        # os.makedirs(test_save_path, exist_ok=True)
                        pass
                    else:
                        test_save_path = None
                    
                    # metric_list = 0.0
                    psnr_list = []
                    ssim_list = []
                    cc_list = []
                    mse_list = []
                    for i_batch, sampled_batch in enumerate(test_load):
                        # pdb.set_trace()
                        D_T2, T2, PD = sampled_batch
                        D_T2, T2, PD = D_T2.cuda(), T2.cuda(), PD.cuda()
                        U_T2 = model(D_T2, PD)

                        avg_score = batch_accessment(T2.cpu().detach().numpy(), U_T2.cpu().detach().numpy(), args.color_range, int(args.scale))#, int(args.scale)
                        # logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
                        # batch_mpsnr= batch_PSNR(T2, U_T2, args.color_range)#, int(args.scale)
                        # batch_mpsnr = compare_mpsnr(T2.cpu().detach().numpy(), U_T2.cpu().detach().numpy(), args.color_range)
                        psnr_list.append(avg_score['MPSNR'])
                        ssim_list.append(avg_score['MSSIM'])
                        cc_list.append(avg_score['CrossCorrelation'])
                        mse_list.append(avg_score['RMSE'])

                        pbar_test.set_postfix_str('test batch:{}, PSNR:{}/{}, SSIM:{}/{}, CC:{}/{}, MSE:{}/{}'.format(i_batch, avg_score['MPSNR'], best_eval_psnr,avg_score['MSSIM'],  best_eval_ssim, avg_score['CrossCorrelation'], best_eval_cc, avg_score['RMSE'], best_eval_mse))
                        pbar_test.update()
                # avg_score = {'MPSNR': 0, 'MSSIM': 0, 'SAM': 0,
                # # 'CrossCorrelation': 0, 'RMSE': 0}
                # mpsnr = avg_score['MPSNR']
                # mssim = avg_score['MSSIM']
                # logging.info('Testing performance in this val model: MPSNR : %f MSSIM : %f  SAM : %f  CrossCorrelation : %f  RMSE : %f' % (avg_score['MPSNR'].item(), avg_score['MSSIM'].item(), avg_score['SAM'].item(), avg_score['CrossCorrelation'].item(), avg_score['RMSE'].item()))
                mpsnr = np.mean(psnr_list)
                mssim = np.mean(ssim_list)
                mcc = np.mean(cc_list)
                mmse = np.mean(mse_list)
                
                # TODO remember best validation metric
                if eval_score_higher_is_better:
                    is_best = mpsnr > best_eval_score
                else:
                    is_best = mpsnr < best_eval_score

                if is_best:
                    # print('Saving new best evaluation metric: %f   mean_dice : %f   mean_hd95 : %f' % (performance, performance, mean_hd95))
                    logging.info('Saving new best evaluation metric: MPSNR : %f MSSIM : %f  CC: %F  MSE: %f' % (mpsnr, mssim, mcc, mmse))
                    best_eval_score = mpsnr
                    best_eval_psnr = mpsnr
                    best_eval_ssim = mssim
                    best_eval_cc = mcc
                    best_eval_mse = mmse

                    # TODO if best save the best model
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save({
                        'model_state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'epoch':epoch_num
                    }, save_mode_path)
                    # print("save the best model to {}".format(save_mode_path))
                    logging.info("save the best model to {}".format(save_mode_path))
                
                pbar_test.set_postfix_str('test epoch:{}, PSNR:{}/{}, SSIM:{}/{}, CC:{}/{}, MSE:{}/{}'.format(epoch_num, mpsnr, best_eval_psnr, mssim,  best_eval_ssim, mcc, best_eval_cc, mmse, best_eval_mse))
                pbar_test.update()
                logging.info('Testing performance in this val model: MPSNR : %f MSSIM : %f  CC: %F  MSE: %f' % (mpsnr, mssim, mcc, mmse))
                    

            model.train()
            # print('The best model of the current process:   mean_dice : %f   mean_hd95 : %f' % (best_eval_dice, best_eval_HD95))
            logging.info('The best model of the current process:   mpsnr : %f ' % (best_eval_score))

        # TODO save model
        save_interval = 10  # int(max_epoch/6)
        # if epoch_num > 1 and (epoch_num + 1) % save_interval == 0:
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({
                        'model_state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'epoch':epoch_num
                    }, save_mode_path)
            # print("save model to {}".format(save_mode_path))
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({
                        'model_state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'epoch':epoch_num
                    }, save_mode_path)
            # print("save model to {}".format(save_mode_path))
            logging.info("save model to {}".format(save_mode_path))
            # iterator.close()
            break
    
    logging.info('The best model of the whole process: MPSNR : %f MSSIM : %f  CC: %F  MSE: %f' % (mpsnr, mssim, mcc, mmse))
    # print('The best model of the whole process:   mean_dice : %f   mean_hd95 : %f' % (best_eval_dice, best_eval_HD95))
    writer.close()
    # print("Training Finished!")
    logging.info("Training Finished!") 
    # os._exit(0)

def run():
    args = build_args()
    main(args)


if __name__=="__main__":
    run()
