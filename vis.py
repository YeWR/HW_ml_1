import os
import numpy as np
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def plot_loss_curve(axix, train_loss, test_loss, file_name='loss.png'):
    plt.title('Loss curve')
    plt.plot(axix, train_loss, color='green', label='training loss')
    plt.plot(axix, test_loss, color='red', label='testing loss')
    # plt.plot(x_axix, train_pn_dis, color='skyblue', label='PN distance')
    # plt.plot(x_axix, thresholds, color='blue', label='threshold')
    plt.legend()  # 显示图例

    plt.xlabel('iteration times')
    plt.ylabel('loss value')
    # plt.show()
    plt.savefig(file_name)
    plt.close()


def plot_curve(axix, arr, title, y_title, file_name='test.png'):
    plt.title(title)
    plt.plot(axix, arr, color='green', label='title')
    plt.legend()  # 显示图例

    plt.xlabel('iteration times')
    plt.ylabel(y_title)
    # plt.show()
    plt.savefig(file_name)
    plt.close()


def acu_curve(y, prob, file_name='roc.png'):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")

    plt.savefig(file_name)
    plt.close()


def plot(vis_data, file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # save data
    for key, val in vis_data.items():
        np.save(os.path.join(file_path, key + '.npy'), val)

    # plot
    train_loss = vis_data['train_loss']
    test_loss = vis_data['test_loss']

    plot_loss_curve(range(len(train_loss)), train_loss, test_loss, file_name=os.path.join(file_path, 'loss.png'))

    AUC = vis_data['auc']
    plot_curve(range(len(AUC)), AUC, 'AUC', 'AUC value', file_name=os.path.join(file_path, 'auc.png'))

    Acc = vis_data['acc']
    plot_curve(range(len(Acc)), Acc, 'Acuracy', 'Acuracy', file_name=os.path.join(file_path, 'acc.png'))

    Y, Y_scores = vis_data['converge_result']
    acu_curve(Y, Y_scores, file_name=os.path.join(file_path, 'roc.png'))