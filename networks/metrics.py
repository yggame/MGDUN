# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   metrics.py
@Time    :   2019/12/4 17:35
@Desc    :
"""
import numpy as np
from scipy.signal import convolve2d
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compare_ergas(x_true, x_pred, ratio):
    """
    Calculate ERGAS, ERGAS offers a global indication of the quality of fused image.The ideal value is 0.
    :param x_true:
    :param x_pred:
    :param ratio: Upsampling scale.
    :return:
    """
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    sum_ergas = 0
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        tmp = r_mse / (np.mean(vec_x)**2)
        sum_ergas += tmp
    return (100 / ratio) * np.sqrt(sum_ergas / x_true.shape[0])


def compare_corr(x_true, x_pred):
    """
    Calculate the cross correlation between x_pred and x_true.
    求对应波段的相关系数，然后取均值
    CC is a spatial measure.
    """
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    x_true = x_true - np.mean(x_true, axis=1).reshape(-1, 1)
    x_pred = x_pred - np.mean(x_pred, axis=1).reshape(-1, 1)
    numerator = np.sum(x_true * x_pred, axis=1).reshape(-1, 1)
    denominator = np.sqrt(np.sum(x_true * x_true, axis=1)
                          * np.sum(x_pred * x_pred, axis=1)).reshape(-1, 1)
    return (numerator / denominator).mean()


def compare_rmse(x_true, x_pred):
    """
    Calculate Root mean squared error
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    return np.linalg.norm(x_true - x_pred) / (np.sqrt(x_true.shape[0] * x_true.shape[1] * x_true.shape[2]))


def compare_mpsnr(x_true, x_pred, data_range, detail=False):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [peak_signal_noise_ratio(image_true=x_true[:, :, k], image_test=x_pred[:, :, k], data_range=data_range)
                  for k in range(channels)]
    if detail:
        return np.mean(total_psnr), total_psnr
    else:
        return np.mean(total_psnr)


def compare_mssim(x_true, x_pred, data_range, multidimension, detail=False):
    """
    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    mssim = [structural_similarity(x_true[:, :, i], x_pred[:, :, i], data_range=data_range, multidimension=multidimension)
             for i in range(x_true.shape[2])]
    if detail:
        return np.mean(mssim), mssim
    else:
        return np.mean(mssim)


def compare_sid(x_true, x_pred):
    """
    SID is an information theoretic measure for spectral similarity and discriminability.
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    N = x_true.shape[2]
    err = np.zeros(N)
    for i in range(N):
        err[i] = abs(np.sum(x_pred[:, :, i] * np.log10((x_pred[:, :, i] + 1e-3) / (x_true[:, :, i] + 1e-3))) +
                     np.sum(x_true[:, :, i] * np.log10((x_true[:, :, i] + 1e-3) / (x_pred[:, :, i] + 1e-3))))
    return np.mean(err / (x_true.shape[1] * x_true.shape[0]))


def compare_appsa(x_true, x_pred):
    """

    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    nom = np.sum(x_true * x_pred, axis=2)
    denom = np.linalg.norm(x_true, axis=2) * np.linalg.norm(x_pred, axis=2)

    cos = np.where((nom / (denom + 1e-3)) > 1, 1, (nom / (denom + 1e-3)))
    appsa = np.arccos(cos)
    return np.sum(appsa) / (x_true.shape[1] * x_true.shape[0])


def compare_mare(x_true, x_pred):
    """

    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    diff = x_true - x_pred
    abs_diff = np.abs(diff)
    # added epsilon to avoid division by zero.
    relative_abs_diff = np.divide(abs_diff, x_true + 1)
    return np.mean(relative_abs_diff)


def img_qi(img1, img2, block_size=8):
    N = block_size ** 2
    sum2_filter = np.ones((block_size, block_size))

    img1_sq = img1 * img1
    img2_sq = img2 * img2
    img12 = img1 * img2

    img1_sum = convolve2d(img1, np.rot90(sum2_filter), mode='valid')
    img2_sum = convolve2d(img2, np.rot90(sum2_filter), mode='valid')
    img1_sq_sum = convolve2d(img1_sq, np.rot90(sum2_filter), mode='valid')
    img2_sq_sum = convolve2d(img2_sq, np.rot90(sum2_filter), mode='valid')
    img12_sum = convolve2d(img12, np.rot90(sum2_filter), mode='valid')

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul
    quality_map = np.ones(denominator.shape)
    index = (denominator1 == 0) & (img12_sq_sum_mul != 0)
    quality_map[index] = 2 * img12_sum_mul[index] / img12_sq_sum_mul[index]
    index = (denominator != 0)
    quality_map[index] = numerator[index] / denominator[index]
    return quality_map.mean()


def compare_qave(x_true, x_pred, block_size=8):
    n_bands = x_true.shape[2]
    q_orig = np.zeros(n_bands)
    for idim in range(n_bands):
        q_orig[idim] = img_qi(x_true[:, :, idim],
                              x_pred[:, :, idim], block_size)
    return q_orig.mean()


def quality_assessment(x_true, x_pred, data_range, ratio, multi_dimension=False):
    """
    :param multi_dimension:
    :param ratio:
    :param data_range:
    :param x_true:
    :param x_pred:
    :param block_size
    :return:
    """
    result = {'MPSNR': compare_mpsnr(x_true=x_true, x_pred=x_pred, data_range=data_range),
              'MSSIM': compare_mssim(x_true=x_true, x_pred=x_pred, data_range=data_range,
                                     multidimension=multi_dimension),
              #   'ERGAS': compare_ergas(x_true=x_true, x_pred=x_pred, ratio=ratio),
            #   'SAM': compare_sam(x_true=x_true, x_pred=x_pred),
              'CrossCorrelation': compare_corr(x_true=x_true, x_pred=x_pred),
              'RMSE': compare_rmse(x_true=x_true, x_pred=x_pred),
              }
    return result


def baseline_assessment(x_true, x_pred, data_range, multi_dimension=False):
    mpsnr, psnrs = compare_mpsnr(x_true=x_true, x_pred=x_pred, data_range=data_range, detail=True)
    mssim, ssims = compare_mssim(x_true=x_true, x_pred=x_pred, data_range=data_range,
                                     multidimension=multi_dimension, detail=True)
    return mpsnr, mssim, psnrs, ssims


def tensor_accessment(x_true, x_pred, data_range, multi_dimension=False):
    x_true = x_true.transpose(0, 2, 3, 1)[0]
    x_pred = x_pred.transpose(0, 2, 3, 1)[0]
    mpsnr, psnrs = compare_mpsnr(x_true=x_true, x_pred=x_pred, data_range=data_range, detail=True)
    mssim, ssims = compare_mssim(x_true=x_true, x_pred=x_pred, data_range=data_range,
                                     multidimension=multi_dimension, detail=True)
    return mpsnr, mssim, psnrs, ssims


def batch_accessment(x_true, x_pred, data_range, ratio, multi_dimension=False):
    scores = []
    avg_score = {'MPSNR': 0, 'MSSIM': 0, 
                 'CrossCorrelation': 0, 'RMSE': 0}
    x_true = x_true.transpose(0, 2, 3, 1)
    x_pred = x_pred.transpose(0, 2, 3, 1)

    for i in range(x_true.shape[0]):
        scores.append(quality_assessment(
            x_true[i], x_pred[i], data_range, ratio, multi_dimension))
    for met in avg_score.keys():
        avg_score[met] = np.mean([score[met] for score in scores])
    return avg_score
