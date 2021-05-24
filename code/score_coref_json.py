# custom coref scoring metric
# a coref cluster is counted as correct iff it exactly matches a gold coref cluster
# this version reads json files with correct answer
# and predicted answer in same file
import sys, glob, json
''' command line arguments '''
input_file=sys.argv[1]
''' end command line arguments '''

def remove_duplicates(l):
    assert isinstance(l,list)
    output=[]
    for x in l:
        if x not in output:
            output.append(x)
    return output

scores={}
for line in open(input_file,'r'):
    data=json.loads(line.strip('\n'))
    gold_clusters=remove_duplicates(data['clusters'])
    predicted_clusters=remove_duplicates(data['predicted_clusters'])
    section=data['subsection']
    scores.setdefault(section,{})
    scores[section].setdefault('num_correct', 0)
    scores[section].setdefault('num_expected',0)
    scores[section]['num_expected']+=len(gold_clusters)
    scores[section].setdefault('num_proposed',0)
    scores[section]['num_proposed']+=len(predicted_clusters)
    for gc in gold_clusters:
        for pc in predicted_clusters:
            match = gc==pc
            if match:
                scores[section]['num_correct'] += 1
                predicted_clusters.remove(pc)
                break
    assert scores[section]['num_correct']<=scores[section]['num_proposed']
    assert scores[section]['num_correct']<=scores[section]['num_expected']
# end for line in open(input_file,'r'):

# compute micro and macro precision, recall and f1
metrics={'micro': {'p':[],'r':[],'f1':[]}, 'macro': {'c':0,'p':0,'e':0}}
for section,numbers in scores.items():
    if numbers['num_proposed']>0:
        metrics['micro']['p'].append(float(numbers['num_correct'])/numbers['num_proposed'])
    if numbers['num_expected']>0:
        metrics['micro']['r'].append(float(numbers['num_correct'])/numbers['num_expected'])
    if (numbers['num_proposed']>0) and (numbers['num_expected']>0):
        metrics['micro']['f1'].append(2*float(numbers['num_correct'])/(numbers['num_expected']+numbers['num_proposed']))
    metrics['macro']['c']+=numbers['num_correct']
    metrics['macro']['p']+=numbers['num_proposed']
    metrics['macro']['e']+=numbers['num_expected']

def average(l):
    return sum(l)/len(l)

def stddev(l):
    avg_square=sum(x**2 for x in l)/len(l)
    square_avg=average(l)**2
    return (avg_square-square_avg)**0.5

def median(l):
    ls=sorted(l)
    k=len(ls)//2
    if len(ls)%2==0:
        median=0.5*(ls[k-1]+ls[k])
    else:
        median=ls[k]
    return median

def accuracy(l):
    return sum([1 if x==1.0 else 0 for x in l])/float(len(l))

def compute_stats(l):
    return {'min': min(l), 'max': max(l), 'avg': average(l), \
            'stddev': stddev(l), 'median': median(l), 'accuracy': accuracy(l)}

metrics['micro']['precision']=compute_stats(metrics['micro']['p'])
metrics['micro']['recall']=compute_stats(metrics['micro']['r'])
metrics['micro']['f1']=compute_stats(metrics['micro']['f1'])
metrics['macro']['precision']=float(metrics['macro']['c'])/metrics['macro']['p']
metrics['macro']['recall']=float(metrics['macro']['c'])/metrics['macro']['e']
metrics['macro']['f1']=2*float(metrics['macro']['c'])/(metrics['macro']['e']+metrics['macro']['p'])

def float2str(x):
    return str(round(100*x,2))

for k1 in ['micro','macro']:
    print(k1)
    for k2 in ['precision','recall','f1']:
        output='\t'+k2+': '
        v=metrics[k1][k2]
        if isinstance(v,float):
            output+=float2str(v)
        elif isinstance(v,dict):
            prefix='\n\t\t'
            output+=prefix+'average +/- stddev:\t' \
                    + float2str(v['avg']) + ' +/- ' + float2str(v['stddev'])
            output+=prefix+'min:\t' + float2str(v['min'])
            output+=prefix+'max:\t' + float2str(v['max'])
            output+=prefix+'median:\t' + float2str(v['median'])
            output+=prefix+'accuracy:\t' + float2str(v['accuracy'])
        print(output)
