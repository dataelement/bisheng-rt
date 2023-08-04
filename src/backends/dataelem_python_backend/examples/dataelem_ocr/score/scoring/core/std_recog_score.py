from .label_replace import replaceFullToHalf

# punctuation = "、·。·`.,;:/(){}?-+~'\"|—*"
punctuation = "、·。·`.,;:/(){}?-+~'\"|—*《》<>…【】〔〕〈〉（）[]_￣ˉ"


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1],
                                           distances_[-1])))
        distances = distances_
    return distances[-1]


def line2kv(x):
    z = x.split(',')
    key = z[0]
    val = ','.join(z[1:])
    val = ''.join(val.strip().split())
    return [key, val]


def to_str(x):
    # only for baidu case
    ans = ''
    for y in x:
        ans += y['words']
    return ''.join(ans.split())


def listify(x):
    ans = []
    i = 0
    while i < len(x):
        tmp = x[i]
        if tmp != '#':
            ans.append(tmp)
            i += 1
        else:
            try:
                tmp_next = x[i + 1]
            except Exception:
                tmp_next = 'a'
            if tmp_next != '#':
                ans.append(tmp)
                i += 1
            else:
                ans.append('##')
                i += 2
    return ans


def depunc(x):
    z = ''
    for xx in x:
        if xx not in punctuation:
            z += xx
    return z


def is_match(s1, s2):
    s1 = s1.replace('##', '卍')
    s2 = s2.replace('##', '卍')
    N1 = len(s1)
    N2 = len(s2)
    if N1 == 0 and N2 == 0:
        return True
    if N1 > 0 and N2 == 0:
        return False
    if N1 == 0 and N2 > 0:
        return all([s == '卍' for s in s2])

    DP = [[False for _ in range(N2)] for _ in range(N1)]
    DP[0][0] = True if s1[0] == s2[0] or s2[0] == '卍' else False
    for i in range(1, N2):
        DP[0][i] = DP[0][i -
                         1] if s2[i] == '卍' or (s2[i - 1] == '卍'
                                                and s1[0] == s2[i]) else False
    for j in range(1, N1):
        for i in range(1, N2):
            if s1[j] == s2[i]:
                DP[j][i] = DP[j - 1][i - 1]
            elif s2[i] == '卍':
                DP[j][i] = DP[j][i - 1] or DP[j - 1][i - 1]
            else:
                DP[j][i] = False
    return DP[-1][-1]


def compare(pred, label, with_punctuation=True, use_dp=False):
    # scoring taking account of the punctuations
    label = ''.join(label.split())
    pred = ''.join(pred.split())
    cer = levenshteinDistance(depunc(replaceFullToHalf(pred)),
                              depunc(replaceFullToHalf(label)))

    if with_punctuation:
        pred = listify(replaceFullToHalf(pred))
        label = listify(replaceFullToHalf(label))
    else:
        pred = listify(depunc(replaceFullToHalf(pred)))
        label = listify(depunc(replaceFullToHalf(label)))

    if use_dp:
        return is_match(''.join(pred), ''.join(label)), cer

    N_pred = len(pred)
    N_label = len(label)
    i = 0
    j = 0
    res = 1
    while i < N_label:
        tmp_label = label[i]
        if j < N_pred:
            tmp_pred = pred[j]
        else:
            tmp_pred = '##'

        if tmp_label != '##':
            if tmp_label == tmp_pred:
                i += 1
                j += 1
            else:
                res = 0
                break
        else:
            spread = 0
            touch_end = False
            while tmp_label == '##' and not touch_end:
                i += 1
                spread += 1
                if i == N_label:
                    touch_end = True
                    continue
                tmp_label = label[i]

            if not touch_end:
                find_flag = False
                for _ in range(spread + 1):
                    if tmp_pred == tmp_label:
                        find_flag = True
                        break
                    j += 1
                    if j >= N_pred:
                        res = 0
                        break
                    tmp_pred = pred[j]

                if not find_flag:
                    res = 0
                    break
            else:
                j += spread
                if j < N_pred:
                    res = 0
                    break

    if res == 0 or j < N_pred:
        return 0, cer
    else:
        return 1, cer
