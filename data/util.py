import numpy as np


def normalize_intensity(img_np, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)): 
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    :param norm_values: (MEAN, STD, MAX, MIN)
    """
    if normalization == "mean":
        mask = img_np[img_np != 0.0]
        desired = img_np[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_np = (img_np - mean_val) / std_val
    
    elif normalization == "max":
        max_val = norm_values[2]
        img_np = img_np / max_val
    
    elif normalization == "max_min":
        img_np = (img_np - norm_values[3]) / (norm_values[2] - norm_values[3])
    
    elif normalization == "full_volume_mean":
        img_np = (img_np - norm_values[0]) / norm_values[1]
    
    elif normalization == 'candi':
        normalized_np = (img_np- norm_values[0]) / norm_values[1]
        final_np = np.where(img_np == 0., img_np, normalized_np)

        final_np = (final_np - np.min(final_np)) / (np.max(final_np) - np.min(final_np))
        x = np.where(img_np == 0., img_np, final_np)
        return x
    
    else:
        img_np = img_np
    
    return img_np


# 区间变换
def unification_interval(data ,interval_min ,interval_max):
    # data         ：需要变换的数据或矩阵
    # interval_min ：变换区间下限。
    # interval_max ：变换区间上限。
    import numpy as np
    data = np.array(data)
    minval = np.min(np.min(data))
    maxval = np.max(np.max(data))

    data = (data -minval ) /(maxval -minval)

    return data *(interval_max -interval_min ) +interval_min