import sys,glob,os, argparse

parser = argparse.ArgumentParser(description='Postprocess rspans')
parser.add_argument('--spans', type=str, required=True,
                    help='path to spans folder')
parser.add_argument('--boundaries', type=str, required=True,
                    help='path to boundaries folder')
parser.add_argument('--candidates', type=str, required=True,
                    help='path to folder containing candidate spans')
args = parser.parse_args()


gold_spans_dir=args.spans.rstrip('/')
boundaries_dir=args.boundaries.rstrip('/')
candidate_spans_dir = args.candidates.rstrip('/')
tolerance = 0 # whether or not to tolerate spans that deviate by some characters
SPLITS=[[68,3301,7703],[1],[151],[152],[2],[63],[3306]]

def get_split_index(section_name):
    section_number=int(section_name[7:])
    output=None
    for ii,x in enumerate(SPLITS):
        if section_number in x:
            assert output is None
            output=ii
    return output

def score_subsection(gold,candidate):
    # precision
    n_correct=0
    for c in candidate:
        correct=False
        for start_offset in range(-tolerance,tolerance+1):
            for end_offset in range(-tolerance,tolerance+1):
                adjusted_span=(c[0]+start_offset,c[1]+end_offset)
                correct = correct or (adjusted_span in gold)
        n_correct+=int(correct)
    precision=(n_correct,len(candidate))
    assert n_correct<=len(candidate)
    # recall
    n_correct=0
    for g in gold:
        correct=False
        for start_offset in range(-tolerance,tolerance+1):
            for end_offset in range(-tolerance,tolerance+1):
                adjusted_span=(g[0]+start_offset,g[1]+end_offset)
                correct = correct or (adjusted_span in candidate)
        n_correct+=int(correct)
    recall=(n_correct,len(gold))
    assert n_correct<=len(gold)
    return precision,recall

def filter_spans(boundaries,spans):
    output=[]
    for s in spans:
        start,end=s
        for b in boundaries:
            start_b,end_b=b
            in_b = (end>=start_b) and (start<=end_b) # ie there is some overlap
            if in_b: # only keep the part of the span that is within the boundaries
                new_start=max(start_b,start)
                new_end=min(end_b,end) 
                output.append((new_start,new_end))
                break
    return output

def score2str(x):
    return str(round(100*x,2))

# check whether we are providing outputs for every section
for filename in glob.glob(gold_spans_dir+'/*'):
    section_name=filename.split('/')[-1]
    assert os.path.isfile(os.path.join(candidate_spans_dir,section_name))
gold_spans,candidate_spans={},{}
boundaries={}
scores={}
for gold_filename in glob.glob(gold_spans_dir+'/*'):
    section_name=gold_filename.split('/')[-1]
    boundaries[section_name]={}
    def read_span(x):
        y=x.strip('\n').split(',')
        return (int(y[0]),int(y[1]))
    def read_gold_spans(filename):
        output={}
        for line in open(filename,'r'):
            if line[0]=='#':
                current_section=line.strip('\n').strip('# ')
                output.setdefault(current_section,set())
            else:
                y=line.strip('\n').split(',')
                output[current_section].add(read_span(line))
        return output
    for x in open(boundaries_dir+'/'+section_name,'r'):
        start,end,name=x.strip('\n').split(',')
        boundaries[section_name].setdefault(name,set())
        boundaries[section_name][name].add((int(start),int(end)))
    gold_spans[section_name]=list(map(read_span, \
            filter(lambda x: x.strip()[0]!='#', open(gold_filename,'r'))))
    candidate_filename=os.path.join(candidate_spans_dir,section_name)
    candidate_spans[section_name]=list(map(read_span, \
            filter(lambda x: x.strip()[0]!='#', open(candidate_filename,'r'))))
scores=[ ((0,0),(0,0)) for _ in SPLITS ]
for section_name in boundaries:
    split_index=get_split_index(section_name)
    n_correct_precision, n_spans_precision, n_correct_recall, n_spans_recall = 0, 0, 0, 0
    for subsection_name in boundaries[section_name]:
        filtered_candidates=filter_spans(boundaries[section_name][subsection_name],\
                    candidate_spans[section_name])
        filtered_gold=filter_spans(boundaries[section_name][subsection_name],\
                    gold_spans[section_name])
        (ncp,nsp),(ncr,nsr)=score_subsection(filtered_gold,filtered_candidates)
        n_correct_precision+=ncp
        n_spans_precision+=nsp
        n_correct_recall+=ncr
        n_spans_recall+=nsr
    (old_ncp, old_nsp), (old_ncr, old_nsr) = scores[split_index]
    scores[split_index] = ( (old_ncp+n_correct_precision,
                             old_nsp+n_spans_precision),
                            (old_ncr+n_correct_recall,
                             old_nsr+n_spans_recall) )

def aggregate_scores(scores):
    n_correct_recall=sum(r[0] for _,r in scores)
    n_gold=sum(r[1] for _,r in scores)
    n_correct_precision=sum(p[0] for p,_ in scores)
    n_candidates=sum(p[1] for p,_ in scores)
    precision=float(n_correct_precision)/n_candidates
    recall=float(n_correct_recall)/n_gold
    f1=2*precision*recall/(precision + recall)
    return precision, recall, f1

macro_precision,macro_recall,macro_f1=aggregate_scores(scores)

print('macro scores')
print('\tprecision:\t' + score2str(macro_precision))
print('\trecall:\t' + score2str(macro_recall))
print('\tf1:\t' + score2str(macro_f1))

def average(l):
    return float(sum(l))/len(l)

def stddev(l):
    avg=average(l)
    stddev=sum((x-avg)**2 for x in l)/len(l)
    stddev=stddev**0.5
    return stddev

def median(l):
    ls=sorted(l)
    k=len(ls)//2
    if len(ls)%2==0:
        median=0.5*(ls[k-1]+ls[k])
    else:
        median=ls[k]
    return median

def compute_stats(l):
    l2=list(filter(lambda x: x is not None, l))
    return {'min': min(l2), 'max': max(l2), 'avg': average(l2), \
                'stddev': stddev(l2), 'median': median(l2)}

def compute_scores(x):
    (n_correct_prec,n_prec),(n_correct_recall,n_recall)=x
    precision=None
    if n_prec>0:
        precision=float(n_correct_prec)/n_prec
    recall=None
    if n_recall>0:
        recall=float(n_correct_recall)/n_recall
    f1=None
    if (recall is not None) and (precision is not None):
        if (precision==0.) or (recall==0.):
            f1=0.
        else:
            f1=2.*precision*recall/(precision+recall)
    return precision,recall,f1

scores=list(map(compute_scores,scores))
print('aggregated scores')
for ii,metric in zip(range(3),['precision','recall','f1']):
    score_list=[v[ii] for v in scores]
    stats=compute_stats(score_list)
    print(metric)
    print('\taverage +/- stddev:\t' + score2str(stats['avg']) + ' +/- ' + score2str(stats['stddev']))
    print('\tmin:\t' + score2str(stats['min']))
    print('\tmax:\t' + score2str(stats['max']))
    print('\tmedian:\t' + score2str(stats['median']))
