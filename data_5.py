# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:10:01 2019

@author: dlymhth
"""

def gen_data(src, tgt, fname):
    with open(src + fname, 'r', encoding='utf-8') as fin, open(tgt + fname, 'w') as fout:
        line = fin.readline()
        fout.write(line)
        lines = fin.readlines()
        fin.close()
        mn_prc, mx_prc = 1000000.0, 0
        lid = 0
        for line in lines:
            f = line.strip().split(',')
            lno = int(f[0]) + 1
            mn_prc = min(mn_prc, float(f[-2]))
            mx_prc = max(mx_prc, float(f[-3]))
            if (lno % 5) == 0:
                f[-2], f[-3], f[0] = str(mn_prc), str(mx_prc), str(lid)
                fout.write(','.join(f) + '\n')
                mn_prc, mx_prc = 1000000.0, 0
                lid += 1


if __name__ == '__main__':
    for v in ['a', 'i', 'j', 'jm', 'm', 'p', 'y']:
        fname = v + '_minutes_clean.csv'
        gen_data('C:/HTH/DFIT/reinforcement/data/',
                  'C:/HTH/DFIT/reinforcement/data_5/', fname)