import sys
sys.path.append('../algo_recommend_server/script/DeepRecommender')

import graph_utils
from sklearn.metrics import ndcg_score

DIRNAME = 'funny/'
TEST_FILENAME = DIRNAME+'test.ratings'
PREDICT_FILENAME = DIRNAME+'predicted_result'


def load_test(filename):
    tests = []
    fin = open(filename, 'r')
    lines = fin.readlines()
    for line in lines:
        params = line.strip().split()
        if len(params) == 3:
            tests.append([int(params[0]), int(params[1]), float(params[2])])
    fin.close()
    return tests


def load_predict(filename):
    predicts = []
    fin = open(filename, 'r')
    lines = fin.readlines()
    for line in lines:
        params = line.strip().split()
        preds = []
        for param in params:
            preds.append(float(param))
        predicts.append(preds)
    fin.close()
    return predicts


def evaluate(tests, predicts, dirname):
    y_test = []
    y_score = []
    for i in range(len(tests)):
        y_test.append(tests[i][2] - 1)
        y_score.append(predicts[i])
    try:
        #print('y_test:', y_test)
        #print('y_score:', y_score)
        graph_utils.f1score(y_test, y_score, step=0.01)
        graph_utils.roc(y_test, y_score, dirname+'preds_roc.png')
        graph_utils.ks(y_test, y_score, dirname+'preds_ks.png')
    except Exception as e:
        print(e)
    
    
def evaluate_ndcg(tests, predicts, dirname):
    y_test = []
    y_score = []
    for i in range(len(tests)):
        y_test.append(tests[i][2] - 1)
        y_score.append(predicts[i])
    print('y_test:', len(y_test))
    print('y_score:', len(y_score))
    score = ndcg_score(y_test, y_score, k=4)  # only for scikit-learn==0.19.0
    print('ndcg_score:', score)

    
if __name__ == '__main__':
    tests = load_test(TEST_FILENAME)
    print('tests:', len(tests))
    predicts = load_predict(PREDICT_FILENAME)
    print('predicts:', len(predicts))
    if len(tests) == len(predicts):
        #evaluate(tests, predicts, DIRNAME)
        evaluate_ndcg(tests, predicts, DIRNAME)
    else:
        print('ERROR!', len(tests), len(predicts))
