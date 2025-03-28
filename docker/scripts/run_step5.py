#!/opt/conda/bin/python
import os, sys, subprocess
from multiprocessing import Pool
def run_cmd(sample,cmd):
    status=subprocess.run(cmd,shell=True).returncode
    if status==0:
        return sample+' succeed'
    else:
        return sample+' fail'

def batch_run(cmds,process_num):
    log=[]
    pool = Pool(processes=process_num)
    result = []
    for cmd in cmds:
        sample=cmd.split()[2]
        process = pool.apply_async(run_cmd,(sample,cmd,))
        result.append(process)
    for process in result:
        log.append(process.get())
    return log




dataset = sys.argv[1]
ncore = int(sys.argv[2])

if not os.path.exists('step5'):
    os.system('mkdir step5/')
if not os.path.exists('step5/' + dataset):
    os.system('mkdir step5/' + dataset)

if os.path.exists('step5_' + dataset + '.cmds'):
    os.system('rm step5_' + dataset + '.cmds')

fp = open(dataset + '_struc.list', 'r')
prots = []
for line in fp:
    words = line.split()
    prots.append(words[0])
fp.close()

need_prots = []
for prot in prots:
    if os.path.exists('step5/' + dataset + '/' + prot + '.result'):
        fp = open('step5/' + dataset + '/' + prot + '.result','r')
        word_counts = set([])
        for line in fp:
            words = line.split()
            word_counts.add(len(words))
        fp.close()
        if len(word_counts) == 1 and 15 in word_counts:
            pass
        else:
            os.system('rm step5/' + dataset + '/' + prot + '.result')
            need_prots.append(prot)
    else:
        if os.path.exists('step5/' + dataset + '/' + prot + '.done'):
            pass
        else:
            need_prots.append(prot)

if need_prots:
    cmds = []
    for prot in need_prots:
        cmds.append('step5_process_hhsearch.py ' + dataset + ' ' + prot)
    logs = batch_run(cmds,ncore)
    fail = [i for i in logs if 'fail' in i]
    if fail:
        with open(dataset + '_step5.log','w') as f:
            for i in fail:
                f.write(i+'\n')
    else:
        with open(dataset + '_step5.log','w') as f:
            f.write('done\n')
else:
    with open(dataset + '_step5.log','w') as f:
        f.write('done\n')
