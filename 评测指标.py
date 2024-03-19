import numpy as np


def calculate_metrics(y_true, y_pred):
    # 计算TP、TN、FP和FN
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    # 计算准确率
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # 计算精确率
    precision = TP / (TP + FP) if TP + FP != 0 else 0

    # 计算召回率
    recall = TP / (TP + FN) if TP + FN != 0 else 0

    # 计算F1值
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return accuracy, precision, recall, f1


# 测试代码
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 0])

accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")