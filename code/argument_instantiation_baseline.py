import sys, json, copy
from argument_instantiation_models import Metrics, get_argument_type
# Simple baselines for argument alignment
# @truth slot: majority label on train
# dollar amounts: find the one number minimizing hinge loss on train
# string: most common answer in train
# Measure metrics on test, with t-test
# Try 2 versions: one where train is gold data, one where train is gold + silver data

""" BEGIN UTIL FUNCTIONS """
def find_minimizer(train_numerical_answers):
    # cost functions
    def relative_error(output,target):
        diff=abs(output-target)
        denom=max(0.1*target,5000)
        return float(diff)/denom

    def hinge_loss(output,target):
        delta=relative_error(output,target)
        return max(delta-1,0)

    def loss(point):
        losses=[hinge_loss(point,target) for target in train_numerical_answers]
        return sum(losses)

    def loss_gradient(point):
        deltas=[relative_error(point,target) for target in train_numerical_answers]
        delta_grad=[1 if d>1 else 0 for d in deltas]
        diffs=[point-target for target in train_numerical_answers]
        diffs_grad=[x/abs(x) if x != 0 else 0 for x in diffs]
        weights=[1./max(5000,0.1*target) for target in train_numerical_answers]
        gradient=sum([a*b*c for a,b,c in zip(delta_grad,diffs_grad,weights)])
        return gradient

    # numerical baseline: minimize convex function with binary search
    def sign(x):
        if x>0:
            return 1
        elif x<0:
            return -1
        elif x==0:
            return 0
        return None

    upper_bound=max(train_numerical_answers)
    lower_bound=min(train_numerical_answers)
    while abs(upper_bound-lower_bound)>1:
        upper_gradient=loss_gradient(upper_bound)
        lower_gradient=loss_gradient(lower_bound)
        middle=round((upper_bound+lower_bound)/2,2)
        middle_gradient=loss_gradient(middle)
        if (sign(lower_gradient)==sign(middle_gradient)) and (sign(lower_gradient)==sign(upper_gradient)):
            assert False
        if middle_gradient==0:
            upper_bound=middle
            lower_bound=middle
        else:
            if sign(upper_gradient)==sign(middle_gradient):
                assert sign(lower_gradient) != sign(middle_gradient)
                upper_bound=middle
            elif sign(lower_gradient)==sign(middle_gradient):
                assert sign(upper_gradient) != sign(middle_gradient)
                lower_bound=middle
            else:
                assert "Problem", (lower_gradient,middle_gradient,upper_gradient)

    best_point=lower_bound
    current_point=lower_bound
    while current_point <= upper_bound:
        current_loss=loss(current_point)
        best_loss=loss(best_point)
        if best_loss>current_loss:
            best_point=current_point
        current_point+=0.01

    return best_point

def find_most_common_element(l):
    counts={}
    for it in l:
        counts.setdefault(str(it),0)
        counts[str(it)]+=1
    counts=sorted(counts.items(),key=lambda x: (x[1],x[0]))
    return counts[-1][0]

# compute parameters of baseline
def make_baseline(data):
    # here data is assumed to be a list of dictionaries
    stats={'@truth': [], 'dollar_amount': [], 'string': []}
    for d in data:
        for k,v in d.items():
            arg_type=get_argument_type(k,arg_value=v)
            assert arg_type in ['binary','span','integer']
            if arg_type=='binary':
                stat_key='@truth'
            elif arg_type=='integer':
                stat_key='dollar_amount'
            else:
                stat_key='string'
            stats[stat_key].append(v)
    for k,v in stats.items():
        assert len(v)>0
    parameters={}
    # pick majority for @truth
    parameters['@truth']=bool(find_most_common_element(stats['@truth']))
    # compute minimizer on numerical answers
    parameters['dollar_amount']=find_minimizer([int(x) for x in stats['dollar_amount']])
    # find most common string
    parameters['string']=find_most_common_element(stats['string'])
    return parameters

# return answer for a specific datapoint
def answer(queries,params):
    output={}
    for q,_ in queries:
        arg_type=get_argument_type(q)
        assert arg_type in ['binary','span','integer']
        if arg_type=='binary':
            stat_key='@truth'
        elif arg_type=='integer':
            stat_key='dollar_amount'
        else:
            stat_key='string'
        output[q]=params[stat_key]
    return output

# evaluate baseline on test set
def eval_baseline(test_data,params):
    metrics=Metrics()
    batch={'predictions': [], 'query_slots': [], 'case_ref': [], 'context': []}
    for d in test_data:
        batch['query_slots'].append(d['query_slots'])
        batch['case_ref'].append(d['case_ref'])
        batch['context'].append(d['case_text'])
        batch['predictions'].append(answer(d['query_slots'],params))
    metrics.accumulate(batch)
    metrics.compute_metrics(compute_confidence_intervals=True)
    return metrics

""" END UTIL FUNCTIONS """

# load data
dataset=sys.argv[1]
data=json.load(open(dataset,'r'))
test_data=[]
for d in data['test']['gold']:
    query_slots=d[3]
    assert isinstance(query_slots,dict)
    query_slots=list(query_slots.items())
    case_ref=d[0]
    case_text=data['cases'][case_ref]
    sample={'query_slots': query_slots, 'case_ref': case_ref, 'case_text': case_text}
    test_data.append(sample)
assert len(test_data)==len(data['test']['gold'])

print('gold data')
gold_data=data['train']['gold']
queries=[]
for x in gold_data:
    slots=x[3]
    assert isinstance(slots,dict)
    queries.append(slots)
parameters=make_baseline(queries)
print(parameters)
metrics=eval_baseline(test_data,parameters)
gold_metrics=metrics.print_metrics()
print(gold_metrics)
print('gold and silver data')
silver_data=data['train']['gold']+data['train']['automatic']
queries=[]
for x in silver_data:
    slots=x[3]
    assert isinstance(slots,dict)
    queries.append(slots)
parameters=make_baseline(queries)
print(parameters)
metrics=eval_baseline(test_data,parameters)
silver_metrics=metrics.print_metrics()
print(silver_metrics)
