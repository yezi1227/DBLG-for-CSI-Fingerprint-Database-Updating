# coding=UTF-8
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process.kernels import RBF
import GPy
import torch
import time
from scipy.signal import savgol_filter
import statsmodels.api as sm
from pykalman import KalmanFilter
from scipy import signal

from deep_bls import DeepBLS
from data_utils import *
from Meet_GAN import build_generator, build_discriminator, build_gan, train_gan

# ========== 文件路径配置 ==========
# A3C模型的状态和奖励文件路径，这些文件是强化学习训练阶段的输出。
stateFile = "state-100_NN50.mat"
rewardFile = "reward-100_NN50.mat"


# ========== 数据加载与预处理函数 ==========
def getOriginalCSI():
    """
    加载原始CSI数据，进行重塑，并收集相应的坐标标签。
    数据从预设路径下的.mat文件中读取。
    """
    xLabel = getXlabel()
    yLabel = getYlabel()
    count = 0
    originalCSI = np.zeros((176, 135000), dtype=np.float32)
    label = np.empty((0, 2), dtype=np.int32)

    for i in range(16):
        for j in range(11):
            filePath = f"D:/DBLG/55SwapData/coordinate{xLabel[i]}{yLabel[j]}.mat"
            if os.path.isfile(filePath):
                c = loadmat(filePath)
                CSI = np.reshape(c['myData'], (1, 3 * 30 * 1500))
                originalCSI[count, :] = CSI
                label = np.append(label, [[int(xLabel[i]), int(yLabel[j])]], axis=0)
                count += 1
    # 返回实际加载的CSI数据和标签。
    return originalCSI[:count, :], label[:count], count


def getXlabel():
    """
    生成X坐标标签列表（从1到16的字符串形式）。
    """
    return [str(i + 1) for i in range(16)]


def getYlabel():
    """
    生成Y坐标标签列表（从01到11的字符串形式，个位数补零）。
    """
    return [f"{j + 1:02d}" if j < 9 else str(j + 1) for j in range(11)]


def generatePilot():
    """
    从原始CSI数据中生成引导（pilot）数据。
    引导数据是原始数据的一个随机选取的子集，用于高斯过程回归的训练。
    """
    originalCSI, label, count = getOriginalCSI()
    originalData = originalCSI[:, 0:2 * 40 * 800:2000].astype(np.float32)
    originalData = SimpleImputer(strategy='mean').fit_transform(originalData)

    rng = np.random.RandomState(20)
    randomLabel = rng.randint(1, count, size=18)
    labelIndex = np.sort(randomLabel)

    listCSI = originalData[labelIndex, :]
    return label[labelIndex], listCSI


def findIndex(label, pathPlan):
    """
    查找给定路径规划中每个坐标点在完整标签列表中的所有对应索引。
    """
    index = []
    for p_x, p_y in pathPlan:
        match_indices = np.where((label[:, 0] == p_x) & (label[:, 1] == p_y))[0]
        if match_indices.size > 0:
            index.extend(match_indices.tolist())
    return index


def filterProcess(mulGauProPrediction, n_iter):
    """
    对CSI预测数据进行卡尔曼滤波和巴特沃斯低通滤波，实现两阶段平滑处理。
    """
    bufferCSI = np.zeros_like(mulGauProPrediction, dtype=np.float32)
    b, a = signal.butter(2, 3 * 2 / 50, 'lowpass')

    for i in range(len(mulGauProPrediction)):
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
        measurements = mulGauProPrediction[i]
        kf = kf.em(measurements, n_iter=n_iter)
        filtered_state_means, _ = kf.filter(measurements)
        finalResult = signal.filtfilt(b, a, filtered_state_means[:, 0])
        bufferCSI[i, :] = finalResult
    return bufferCSI


def isOdd(n):
    """
    判断一个数是否为奇数。如果不是奇数，则返回比它小1的奇数。
    用于确保Savitzky-Golay滤波器窗口大小为奇数。
    """
    n_int = int(n)
    return n_int if n_int % 2 == 1 else n_int - 1


def find_close_fast(arr, errorBand):
    """
    在数组 `arr` 中找到一个元素，使其与 `errorBand` 中随机选择的一个元素最接近。
    此函数用于动态确定Savitzky-Golay滤波器的滑动窗口大小。
    """
    low = 0
    high = len(arr) - 1
    idx = -1
    rng = np.random.RandomState(20)
    randomInt = rng.randint(len(errorBand[0]))
    target_val = errorBand[randomInt, 0]

    while low <= high:
        mid = (low + high) // 2
        if target_val == arr[mid] or mid == low:
            idx = mid
            break
        elif target_val > arr[mid]:
            low = mid
        elif target_val < arr[mid]:
            high = mid
    if idx + 1 < len(arr) and abs(target_val - arr[idx]) > abs(target_val - arr[idx + 1]):
        idx += 1

    return arr[idx]


def findPossiblePath(stateFile):
    """
    从A3C模型的状态文件中加载所有可能的路径，并筛选出包含特定起点([1, 1])和终点([16, 11])的路径。
    """
    possiblePath = []
    stateLabel = []
    state = loadmat(f"D:/DBLG/Fifth code Meet/{stateFile}")
    stateList = np.reshape(state['array'], (100, 100, 2))

    for i in range(100):
        a = stateList[i].tolist()
        new_list = [list(t) for t in set(tuple(xx) for xx in a)]
        new_list.sort()

        if [1, 1] in new_list and [16, 11] in new_list:
            possiblePath.append(new_list)
            stateLabel.append(i)
    return possiblePath, stateLabel


def OptimalPath(rewardFile):
    """
    根据A3C算法学习到的奖励值，从所有可能的路径中选择具有最高奖励值（即最优）的路径。
    """
    possiblePath, stateLabel = findPossiblePath(stateFile)
    reward = loadmat(f"D:/DBLG/Fifth code Meet/{rewardFile}")
    rewardList = reward['array'][0]

    valueOfReward = [rewardList[idx] for idx in stateLabel]
    max_index = np.argmax(np.array(valueOfReward))
    optimalPath = possiblePath[int(max_index)]
    return optimalPath, np.max(valueOfReward)


# ========== 精度评估函数 ==========
def accuracyPre(predictions, labels):
    """
    计算预测位置与真实标签之间欧几里得距离的平均值（平均定位误差），并将其转换为米。
    """
    errors = np.sqrt(np.sum((predictions - labels) ** 2, axis=1))
    return np.mean(errors) * 60 / 100


def accuracyStd(predictions, testLabel):
    """
    计算预测位置与真实标签之间欧几里得距离的标准差，并将其转换为米。
    """
    error = np.asarray(predictions - testLabel)
    sample = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2) * 60 / 100
    return np.std(sample)


def saveTestErrorMat(predictions, testLabel, fileName):
    """
    将测试误差（欧几里得距离，单位：米）保存到MATLAB .mat 文件中。
    """
    error = np.asarray(predictions - testLabel)
    sample = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2) * 60 / 100
    savemat(fileName + '.mat', {'array': sample})

def adjusted_generate(raw_data, generated_data):
    original_mean = np.mean(raw_data)
    original_std = np.std(raw_data)
    generated_mean = np.mean(generated_data)
    generated_std = np.std(generated_data)
    return (generated_data - generated_mean) / generated_std * original_std + original_mean


# ========== 主程序入口 ==========
if __name__ == '__main__':
    # 设置TensorFlow的日志级别为“只显示错误”，避免不必要的警告信息。
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # --- 1. 原始CSI数据加载与引导数据生成 ---
    originalCSI, label, count = getOriginalCSI()
    pilotLabel, pilotCSI = generatePilot()

    # --- 2. 多元高斯过程回归（GPR）建模与预测 ---
    mean = np.mean(pilotLabel, axis=1)
    covMatrix = np.cov(pilotCSI)
    kernelRBF = GPy.kern.RBF(input_dim=2, variance=1)
    omiga = kernelRBF.K(pilotLabel, pilotLabel)
    kernelPilot = covMatrix * omiga
    np.random.seed(0)
    mulGauProPrediction = np.random.multivariate_normal(mean, kernelPilot, size=len(label))

    # --- 3. 滤波平滑处理 ---
    bufferCSI = filterProcess(mulGauProPrediction, n_iter=2)

    # --- 4. 状态空间模型修正（误差带估计） ---
    meanError = np.mean(pilotCSI, axis=0)
    newModel = sm.tsa.SARIMAX(meanError, order=(1, 0, 0), trend='c')
    results = newModel.fit(disp=False)
    predict_sari = results.get_prediction()
    errorBand = predict_sari.conf_int()

    # --- 5. 误差带约束与Savitzky-Golay平滑修正 ---
    filterMatrix = bufferCSI
    for i in range(len(bufferCSI)):
        sliding_window = isOdd(find_close_fast(bufferCSI[i], errorBand))
        tmp_result = savgol_filter(bufferCSI[i], sliding_window, 2)
        filterMatrix[i, :] = tmp_result

    # --- 6. A3C路径规划 ---
    pathPlan, maxReward = OptimalPath(rewardFile)

    # --- 7. DeepBLS和GAN数据子集准备与清洗 ---
    index_A3CPredict = np.array(findIndex(label, pathPlan)).flatten()
    index = np.array(findIndex(label, pilotLabel)).flatten()
    index_GaussianAndA3C = np.sort(list(set(np.append(index_A3CPredict, index, axis=0))))

    secondpilotCSI = originalCSI[index_GaussianAndA3C,]

    secondpilotCSI_cl_df = pd.DataFrame(secondpilotCSI)
    secondpilotCSI_cl_df.replace(0, np.nan, inplace=True)
    secondpilotCSI_cl_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    mean_values = secondpilotCSI_cl_df.mean()
    secondpilotCSI_cl_df.fillna(mean_values, inplace=True)
    secondpilotCSI_cl_processed = secondpilotCSI_cl_df.values

    corresponding_labels = label[index_GaussianAndA3C,]
    secondpilotCSI_cl_processed32 = np.array(secondpilotCSI_cl_processed[:, 0:2 * 40 * 800:2000],
                                             dtype=np.float32)

    # --- 8. DeepBLS 模型训练与预测 ---
    train_x, test_x, train_y, test_y = train_test_split(
        secondpilotCSI_cl_processed32, corresponding_labels, test_size=0.2, random_state=10
    )
    model = DeepBLS(max_iter=50, learn_rate=0.01, new_BLS_max_iter=10,
                    NumFea=20, NumWin=5, NumEnhan=50, s=0.8, C=2 ** -30)
    model.fit(train_x, train_y)
    final_predictions = model.predict(originalCSI[:, 0:2 * 40 * 800:2000].astype(np.float32))

    # --- 9. GAN 模型训练与数据生成 ---
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    np.random.seed(1)

    gauss_inputs = secondpilotCSI_cl_processed
    gauss_label = corresponding_labels

    stacked_arrays = np.concatenate((label, gauss_label), axis=0)
    unique_rows, counts = np.unique(stacked_arrays, axis=0, return_counts=True)
    remainder_label = unique_rows[counts == 1]

    latent_dim = 1000
    data_dim = gauss_inputs.shape[1]
    label_dim = gauss_label.shape[1]

    generator = build_generator(latent_dim, label_dim, data_dim)
    discriminator = build_discriminator(data_dim, label_dim)
    gan = build_gan(generator, discriminator, latent_dim=latent_dim, label_dim=label_dim)
    train_gan(generator, discriminator, gan, gauss_inputs, gauss_label, epochs=50, batch_size=32)

    noise = np.random.normal(0, 1, (len(remainder_label), latent_dim))
    generated_data = generator.predict([noise, remainder_label])
    adjusted_generated_data = adjusted_generate(gauss_inputs, generated_data)

    # --- 10. 数据合并与最终指纹图融合 ---
    merged_labels = np.concatenate((gauss_label, remainder_label), axis=0)
    merged_data = np.concatenate((gauss_inputs, adjusted_generated_data), axis=0)

    sorted_indices = np.lexsort((merged_labels[:, 1], merged_labels[:, 0]))
    sort_data = merged_data[sorted_indices]

    GAN_data = sort_data.reshape((count, 3 * 30 * 1500))

    GAN_data32 = GAN_data[:, 0:2 * 40 * 800:2000].astype(np.float32)
    thirdFinger = 0.4 * GAN_data32 + 0.6 * final_predictions

    # --- 11. KNN定位测试与性能评估 ---
    traindata1, testdata1, trainlabel1, testlabel1 = train_test_split(
        thirdFinger, label, test_size=0.2, random_state=3
    )

    knn_model = KNeighborsRegressor(n_neighbors=12).fit(traindata1, trainlabel1)

    time_start = time.time()
    prediction = knn_model.predict(testdata1)
    prediction_time_elapsed = time.time() - time_start

    print(f"平均定位误差: {accuracyPre(prediction, testlabel1):.2f} m")
    print(f"定位误差标准差: {accuracyStd(prediction, testlabel1):.2f} m")
    print(f"预测时间: {prediction_time_elapsed:.2f} s")

    # 保存测试误差到.mat文件。
    saveTestErrorMat(prediction, testlabel1, 'Predict-Meet-Error')