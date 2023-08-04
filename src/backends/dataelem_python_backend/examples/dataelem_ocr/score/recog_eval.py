try:
    from .scoring.score import score
except Exception:  # ImportError
    from scoring.score import score

import argparse
import json


def main(label_path, submission_path):
    # path of your submission file
    # submission file json format {'imgname' : {'value' : ['prediction']}}

    save_path = submission_path.replace('.txt', '_convert.txt')
    memo = {}
    with open(submission_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(':', 1)
            memo[line[0]] = {'value': [line[1]]}
    with open(save_path, 'w') as f:
        json.dump(memo, f, ensure_ascii=False)

    # mode supports [baidu, normal], we use normal for own algorithm results
    mode = 'normal'
    # if image label has the tags in exclude_tags, this image will not count
    # for the scoring process
    exclude_tags = ['无法辨认', '可过滤']

    df, df_rare, df_CN_TW = score(save_path,
                                  label_path,
                                  mode,
                                  exclude_tags,
                                  use_dp=True)
    return float(df[-1:]['acc_punc']), float(df[-1:]['acc_depunc'])


if __name__ == '__main__':
    # path of your submission file
    # submission file json format {'imgname' : {'value' : ['prediction']}}
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_path',
                        default=None,
                        type=str,
                        required=True,
                        help='predict')
    parser.add_argument('--label_path',
                        default=None,
                        type=str,
                        required=True,
                        help='label')
    args = parser.parse_args()

    acc_punc, acc_depunc = main(args.label_path, args.submission_path)
    print('acc_punc:{}, acc_depunc:{}'.format(acc_punc, acc_depunc))
