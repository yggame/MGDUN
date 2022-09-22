from torch import from_numpy
from tqdm import tqdm

from .networks.metrics import *


def inference(args, model, testloader, logging, test_save_path=None):
    # print("{} test iterations per epoch".format(len(testloader)))
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # pdb.set_trace()
        D_T2, T2, PD = sampled_batch
        with torch.no_grad():
            U_T2 = model(D_T2)
        metric_i = test_single_volume(D_T2, T2, PD, model)
        metric_list += np.array(metric_i)
        # print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(testloader)
    for i in range(1, args.num_classes):
        # print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    # print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return performance, mean_hd95, "Testing Finished!"