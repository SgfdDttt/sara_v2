import sys, glob
exp_dir=sys.argv[1]
results={}

metric_names=['f1','precision','recall','f1-5','precision-5','recall-5']
for key in ['dev','test']:
    results[key]={}
    for metric in metric_names:
        results[key][metric]=[]

for filename in glob.glob(exp_dir+'/split*/best_model_info'):
    stuff=list(open(filename,'r'))[0]
    recall=float(stuff.split('dev recall ')[1].split(';')[0])
    precision=float(stuff.split('dev precision ')[1].split(';')[0])
    f1=float(stuff.split('dev f1 ')[1].split(';')[0])
    results['dev']['recall'].append(recall)
    results['dev']['precision'].append(precision)
    results['dev']['f1'].append(f1)
    recall=float(stuff.split('dev recall-5 ')[1].split(';')[0])
    precision=float(stuff.split('dev precision-5 ')[1].split(';')[0])
    f1=float(stuff.split('dev f1-5 ')[1].split(';')[0])
    results['dev']['recall-5'].append(recall)
    results['dev']['precision-5'].append(precision)
    results['dev']['f1-5'].append(f1)

for filename in glob.glob(exp_dir+'/split*/best_model_test_results'):
    stuff=list(open(filename,'r'))[0]
    recall=float(stuff.split('recall ')[1].split(';')[0])
    precision=float(stuff.split('precision ')[1].split(';')[0])
    f1=float(stuff.split('f1 ')[1].split(';')[0])
    results['test']['recall'].append(recall)
    results['test']['precision'].append(precision)
    results['test']['f1'].append(f1)
    recall=float(stuff.split('recall-5 ')[1].split(';')[0])
    precision=float(stuff.split('precision-5 ')[1].split(';')[0])
    f1=float(stuff.split('f1-5 ')[1].split(';')[0])
    results['test']['recall-5'].append(recall)
    results['test']['precision-5'].append(precision)
    results['test']['f1-5'].append(f1)

def mean(l):
    return float(sum(l))/len(l)

def stddev(l):
    m=mean(l)
    l2=list((x-m)**2 for x in l)
    stddev=mean(l2)
    stddev=stddev**0.5
    return stddev

def compute_stats(l):
    return {'min': min(l), 'max': max(l), 'mean': mean(l), 'stddev': stddev(l)}

for key in ['dev','test']:
    print(key)
    for metric in metric_names:
        print(metric)
        print(compute_stats(results[key][metric]))

