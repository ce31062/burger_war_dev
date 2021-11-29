import numpy as np

def lidar_transform(lidar, debug=True):

    if debug:
        if min(lidar) < 0.15:
            idx = np.where(lidar > 3.5)
            lidar[idx] = 0.12
        else:
            idx = np.where(lidar > 3.5)
            lidar[idx] = 3.5
    else:
        zero_idx = []
        for i in range(len(lidar)):
            if lidar[i] < 0.01:
                zero_idx.append(i)
            else:
                if len(zero_idx) != 0:
                    if zero_idx[0] > 1:
                        min_idx = zero_idx[0] - 1
                    else:
                        min_idx = 359
                        while lidar[min_idx] < 0.01:
                            zero_idx.append(min_idx)
                            min_idx -= 1
                            
                    avg = lidar[[min_idx, i]].mean()
                    lidar[zero_idx] = avg

                    zero_idx = []

    return lidar
