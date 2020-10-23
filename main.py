import numpy as np
from sklearn.metrics import roc_auc_score


def transform(line, feature_dim):
    data = line.split(' ')
    label = int(data[0])
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
    weights = np.random.rand(feature_dim, 1) * (0.)
    return weights


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


def IRLS_update(ld, W, X, Y):
    MU, R = MU_R(W, X)
    try:
        matrix = np.matrix(ld + X.T.dot(R).dot(X)).I
    except:
        matrix = np.linalg.pinv(np.matrix(ld + X.T.dot(R).dot(X)))
        import ipdb
        ipdb.set_trace()
    delta = matrix.dot(- ld * W + X.T.dot(MU - Y))
    return W - 1e-4 * delta


def Loss(ld, W, X, Y):
    loss = 0.
    for (xi, yi) in zip(X, Y):
        loss += (yi * W.T.dot(xi) - np.log(1. + np.exp(W.T.dot(xi)))).item()

    loss = loss - ld / 2 * W.T.dot(W).item()
    return loss


def train(config):
    W = init_weights(config['feature_dim'])
    ld, X, Y = config['ld'], config['train_features'], config['train_labels']
    epsilon = 1e-3
    loss = np.inf
    epoch = 1
    while np.abs(loss) >= epsilon:
        train_loss = - Loss(ld, W, X, Y)
        test_loss, auc = test(config, W)
        print('Epoch {:<5d}: training loss: {:<10f}; testing loss: {:<10f}, AUC: {:<4f}'.format(epoch, train_loss, test_loss, auc))

        W = IRLS_update(ld, W, X, Y)

        epoch += 1


def test(config, W):
    ld, X, Y = config['ld'], config['test_features'], config['test_labels']
    loss = - Loss(ld, W, X, Y)

    Y_true = Y.flatten()
    Y_scores = []
    for xi in X:
        score = mu(W, xi)
        Y_scores.append(score.item())

    auc = roc_auc_score(Y_true, Y_scores)

    return loss, auc


if __name__=='__main__':
    config = {}

    config['feature_dim'] = 123
    config['train_file'] = 'a9a'
    config['test_file'] = 'a9a.t'
    config['ld'] = 0.

    config['train_labels'], config['train_features'] = load_data(config['train_file'], config['feature_dim'])
    config['test_labels'], config['test_features'] = load_data(config['test_file'], config['feature_dim'])

    train(config)

