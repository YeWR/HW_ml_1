from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from vis import plot

def transform(line, feature_dim):
    data = line.split(' ')
    label = int(data[0])
    if label < 0:
        label = 0
    feature = np.zeros(feature_dim)
    for dt in data[1:]:
        feature_id, val = dt.split(':')
        feature_id, val = int(feature_id), int(val)
        feature[feature_id - 1] = val

    return label, feature


def load_data(file_path, feature_dim):
    labels = []
    features = []

    for line in open(file_path):
        label, feature = transform(line.strip(), feature_dim)
        labels.append(label)
        features.append(feature)

    num = len(labels)
    labels = np.array(labels).reshape(num, 1)
    features = np.array(features)
    return labels, features


def init_weights(feature_dim):
    weights = np.random.randn(feature_dim, 1) * 1e-2
    return weights


def adjust_lr(init_lr, step, total_epochs):
    lr = init_lr * (0.1 ** (step / total_epochs))
    return lr


def sigmoid(x):
    return 1. / (1. + np.exp(- x))


def mu(weights, feature):
    return sigmoid(weights.T.dot(feature).item())


def MU_R(W, X):
    feature_num, _ = X.shape
    MU = np.zeros(feature_num)
    R = np.zeros((feature_num, feature_num))

    for i in range(feature_num):
        MU[i] = mu(W, X[i])
        R[i][i] = MU[i] * (1. - MU[i])

    return MU.reshape(feature_num, 1), R


def IRLS_update(lr, ld, W, X, Y):
    MU, R = MU_R(W, X)
    matrix = np.linalg.pinv(np.matrix(ld * np.identity(X.shape[1]) + X.T.dot(R).dot(X)))
    gradient = matrix.dot(ld * W + X.T.dot(MU - Y)) # .clip(-1., 1.)
    return W - lr * gradient


def Loss(ld, W, X, Y):
    loss = 0.
    for (xi, yi) in zip(X, Y):
        temp = W.T.dot(xi)
        loss += (yi * temp - np.log(1. + np.exp(temp))).item()

    loss = loss - ld / 2 * np.linalg.norm(W)
    return loss


def train(config):
    W = init_weights(config['feature_dim'])
    lr, ld, X, Y = config['lr'], config['ld'], config['train_features'], config['train_labels']
    total_epochs = config['total_epochs']
    Y_scores = []

    vis_data = {
        'train_loss':[],
        'test_loss': [],
        'auc': [],
        'acc': [],
        'train_acc': [],
        'lr': [],
        'norm': [],
    }

    for epoch in tqdm(range(total_epochs)):
        _, train_loss, _, train_acc = test(config, W, X, Y)
        Y_scores, test_loss, auc, acc = test(config, W, config['test_features'], config['test_labels'])
        norm = W.T.dot(W).item()
        print('Epoch {:<5d}: Lr: {:<10f}; training loss: {:<10f}, Acuracy: {:<4f}; testing loss: {:<10f}, AUC: {:<4f}, Acuracy: {:<4f}; L2-norm: {:<4f}'.format(epoch, lr, train_loss, train_acc, test_loss, auc, acc, norm))

        W = IRLS_update(lr, ld, W, X, Y)

        # lr = adjust_lr(config['lr'], epoch, total_epochs // 3)

        vis_data['train_loss'].append(train_loss)
        vis_data['test_loss'].append(test_loss)
        vis_data['auc'].append(auc)
        vis_data['acc'].append(acc)
        vis_data['train_acc'].append(train_acc)
        vis_data['lr'].append(lr)
        vis_data['norm'].append(norm)

    vis_data['converge_result'] = [config['test_labels'].flatten(), Y_scores]
    plot(vis_data, file_path='vis/ld_' + str(ld))


def test(config, W, X, Y):
    ld = config['ld']
    loss = - Loss(ld, W, X, Y)

    Y_true = Y.flatten()
    Y_scores = []
    acc = 0
    for xi, yi in zip(X, Y_true):
        score = mu(W, xi)
        Y_scores.append(score.item())

        if (score >= 0.5 and yi == 1) or (score < 0.5 and yi == 0):
            acc += 1

    acc /= len(X)

    auc = roc_auc_score(Y_true, Y_scores)

    return Y_scores, loss, auc, acc


if __name__=='__main__':
    config = {}

    config['feature_dim'] = 123
    config['train_file'] = 'a9a'
    config['test_file'] = 'a9a.t'
    config['ld'] = 0
    config['lr'] = 1
    config['total_epochs'] = 20

    config['train_labels'], config['train_features'] = load_data(config['train_file'], config['feature_dim'])
    config['test_labels'], config['test_features'] = load_data(config['test_file'], config['feature_dim'])

    train(config)