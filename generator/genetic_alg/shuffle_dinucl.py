#!/usr/bin/env
import numpy as np

LETTERS = ["A", "C", "G", "T"]


def shuffle_seq(s):
    chars = list(s)
    np.random.shuffle(chars)
    return "".join(chars)


def compute_count(s):
    # P. Clote, Oct 2003
    # Initialize lists and mono- and dinucleotide dictionaries
    dct = {i: list() for i in LETTERS}
    nucl_list = list(LETTERS)
    nucl_cnt = dict()
    dinucl_cnt = dict()
    for x in nucl_list:
        nucl_cnt[x] = 0
        dinucl_cnt[x] = {}
        for y in nucl_list:
            dinucl_cnt[x][y] = 0

    nucl_cnt[s[0]] = 1
    nucl_total = 1
    dinucl_total = 0
    for i in range(len(s) - 1):
        x = s[i]
        y = s[i + 1]
        dct[x].append(y)
        nucl_cnt[y] += 1
        nucl_total += 1
        dinucl_cnt[x][y] += 1
        dinucl_total += 1
    assert nucl_total == len(s)
    assert dinucl_total == len(s) - 1
    return dinucl_cnt, dct


def choose_edge(x, dinucl_cnt):
    # P. Clote, Oct 2003
    z = np.random.random()
    denom = dinucl_cnt[x]['A'] + dinucl_cnt[x]['C'] + dinucl_cnt[x]['G'] + dinucl_cnt[x]['T']
    numerator = dinucl_cnt[x]['A']
    if z < numerator / denom:
        dinucl_cnt[x]['A'] -= 1
        return 'A'
    numerator += dinucl_cnt[x]['C']
    if z < numerator / denom:
        dinucl_cnt[x]['C'] -= 1
        return 'C'
    numerator += dinucl_cnt[x]['G']
    if z < numerator / denom:
        dinucl_cnt[x]['G'] -= 1
        return 'G'
    dinucl_cnt[x]['T'] -= 1
    return 'T'


def connected_to_last(edge_list, nucl_list, last_char):
    # P. Clote, Oct 2003
    dct = {x: 0 for x in nucl_list}
    for edge in edge_list:
        a = edge[0]
        b = edge[1]
        if b == last_char:
            dct[a] = 1
    for i in range(2):
        for edge in edge_list:
            a = edge[0]
            b = edge[1]
            if dct[b] == 1:
                dct[a] = 1
    for x in nucl_list:
        if x != last_char and dct[x] == 0:
            return False
    return True


def eulerian(s):
    # P. Clote, Oct 2003
    dinucl_cnt, dct = compute_count(s)
    nucl_list = []
    for x in LETTERS:
        if x in s:
            nucl_list.append(x)

    last_char = s[-1]
    edge_list = []
    for x in nucl_list:
        if x != last_char:
            edge_list.append((x, choose_edge(x, dinucl_cnt)))
    ok = connected_to_last(edge_list, nucl_list, last_char)
    return ok, edge_list, nucl_list


def shuffle_edge_list(lst):
    # P. Clote, Oct 2003
    n = len(lst)
    barrier = n
    for i in range(n - 1):
        z = int(np.random.random() * barrier)
        lst[z], lst[barrier - 1] = lst[barrier - 1], lst[z]
        barrier -= 1
    return lst


def shuffle_seq_dinucl(s):
    # P. Clote, Oct 2003
    ok = False
    while not ok:
        ok, edge_list, nucl_list = eulerian(s)
    dinucl_cnt, dct = compute_count(s)

    # remove last edges from each vertex list, shuffle, then add back
    # the removed edges at end of vertex lists.
    for x, y in edge_list:
        dct[x].remove(y)
    for x in nucl_list:
        shuffle_edge_list(dct[x])
    for x, y in edge_list:
        dct[x].append(y)

    lst = [s[0]]
    prev_char = s[0]
    for i in range(len(s)-2):
        char = dct[prev_char][0]
        lst.append(char)
        del dct[prev_char][0]
        prev_char = char
    lst.append(s[-1])
    return "".join(lst)