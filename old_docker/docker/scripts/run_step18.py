#!/opt/conda/bin/python
import os, sys, subprocess
from multiprocessing import Pool

def run_cmd(cmd):
    status=subprocess.run(cmd,shell=True).returncode
    if status==0:
        return cmd+' succeed'
    else:
        return cmd+' fail'

def batch_run(cmds,process_num):
    log=[]
    pool = Pool(processes=process_num)
    result = []
    for cmd in cmds:
        process = pool.apply_async(run_cmd,(cmd,))
        result.append(process)
    for process in result:
        log.append(process.get())
    return log



dataset = sys.argv[1]
ncore = int(sys.argv[2])
if not os.path.exists('step18'):
    os.system('mkdir step18/')
if not os.path.exists('step18/' + dataset):
    os.system('mkdir step18/' + dataset)

fp = open(dataset + '_struc.list', 'r')
prots = []
for line in fp:
    words = line.split()
    prots.append(words[0])
fp.close()

need_prots = []
for prot in prots:
    if os.path.exists('step18/' + dataset + '/' + prot + '.data'):
        word_counts = set([])
        fp = open('step18/' + dataset + '/' + prot + '.data', 'r')
        for line in fp:
            words = line.split()
            word_counts.add(len(words))
        fp.close()
        if len(word_counts) == 1 and 8 in word_counts:
            pass
        else:
            os.system('rm step18/' + dataset + '/' + prot + '.data')
            need_prots.append(prot)
    elif os.path.exists('step18/' + dataset + '/' + prot + '.done'):
        pass
    else:
        need_prots.append(prot)

if need_prots:
    cmds = []
    for prot in need_prots:
        cmds.append('step18_get_mapping.py ' + dataset + ' ' + prot)
    logs = batch_run(cmds, ncore)
    fail = [i for i in logs if 'fail' in i]
    if fail:
        with open(dataset + '_step18.log','w') as f:
            for i in fail:
                f.write(i+'\n')
    else:
        with open(dataset + '_step18.log','w') as f:
            f.write('done\n')
else:
    with open(dataset + '_step18.log','w') as f:
        f.write('done\n')
