import json, random, sys

random.seed(17)
datafile=sys.argv[1]
output_file=sys.argv[2]
num_splits=10

def to_json(x):
    output=[]
    # first 2 fields are strings
    output.append(x[0])
    output.append(x[1])
    # third field is a dictioanry
    input_dict=x[2]
    output.append(input_dict)
    # fourth field is a dict
    outcome=x[3]
    output.append(outcome)
    return output

data=json.load(open(datafile,'r'))
train_data=data['train']['gold']
case_ids={}
case_ids['binary']=list(filter(lambda x: not x[0].startswith('tax_case'), train_data))
case_ids['numerical']=list(filter(lambda x: x[0].startswith('tax_case'), train_data))
assert len(case_ids['binary'])+len(case_ids['numerical'])==len(train_data)
case_ids['binary']=set(x[0][:-4] for x in case_ids['binary'])
case_ids['numerical']=set(x[0] for x in case_ids['numerical'])
assert len(case_ids['binary'])*2+len(case_ids['numerical'])==len(train_data)
splits=[set() for _ in range(num_splits)]
# shuffle
case_ids['binary']=sorted(case_ids['binary']); random.shuffle(case_ids['binary'])
case_ids['numerical']=sorted(case_ids['numerical']); random.shuffle(case_ids['numerical'])
split_id=0
while len(case_ids['binary'])>0:
    case_id=case_ids['binary'].pop()
    splits[split_id].add(case_id)
    split_id = (split_id+1) % num_splits
while len(case_ids['numerical'])>0:
    case_id=case_ids['numerical'].pop()
    splits[split_id].add(case_id)
    split_id = (split_id+1) % num_splits
split_index={}
for spliti,split in enumerate(splits):
    for y in split:
        split_index[y]=spliti
output=[]
for ii,x in enumerate(data['train']['gold']):
    case_id=x[0]
    if not case_id.startswith('tax_case'):
        case_id=case_id[:-4]
    spliti=split_index[case_id]
    output.append(to_json(x)+[spliti])
output={'train': output, 'test': data['test']['gold']}
for key in data:
    output.setdefault(key, data[key])
json.dump(output,open(output_file,'w'),indent=2,sort_keys=True)
