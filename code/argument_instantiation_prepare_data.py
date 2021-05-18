import glob, json, argparse

parser = argparse.ArgumentParser(description='Prepare data for argument instantiation')
parser.add_argument('--statutes', type=str, required=True,
                    help='path to statutes folder')
parser.add_argument('--structure', type=str, required=True,
                    help='path to folder containing structure annotations')
parser.add_argument('--cases', type=str, required=True,
                    help='path to folder containing cases')
parser.add_argument('--splits', type=str, required=True,
                    help='path to folder containing data splits for cases')
parser.add_argument('--spans', type=str, required=True,
                    help='path to span folder')
parser.add_argument('--boundaries', type=str, required=True,
                    help='path to boundaries folder')
parser.add_argument('--gold_argument_instantiation', type=str, required=True,
                    help='path to argument instantiation annotations')
parser.add_argument('--silver_argument_instantiation', type=str, required=True,
                    help='path to argument instantiation data generated with Prolog program')
parser.add_argument('--savefile', type=str, required=True,
                    help='name of file to save data in')
args = parser.parse_args()


# preprocess slot filling data and boundaries
output_file=args.savefile
# statute and case files
statutes_dir=args.statutes.rstrip('/')
case_dir=args.cases.rstrip('/')
splits_dir=args.splits.rstrip('/')
# span annotations
spans_dir=args.spans.rstrip('/')
# structure annotations
boundaries_dir=args.boundaries.rstrip('/')
structure_dir=args.structure.rstrip('/')
# slot filling annotations
gold_slot_filling_annotations=args.gold_argument_instantiation
automatic_slot_filling_annotations=args.silver_argument_instantiation

# build mapping of case name -> context
case2txt={}
for filename in glob.glob(case_dir+'/*'):
    casename=filename.split('/')[-1][:-3]
    stuff=[line.strip('\n') for line in open(filename,'r')]
    text_index=stuff.index('% Text')
    question_index=stuff.index('% Question')
    case2txt[casename]=' '.join(x.strip('% ') \
            for x in stuff[text_index+1:question_index-1])
    case2txt[casename]=case2txt[casename].strip(' ')
    inference_hypothesis=stuff[question_index+1].lstrip('% ')
    inference_hypothesis=' '.join(inference_hypothesis.split(' ')[:-1]).lstrip('% ')
    case2txt[casename]+=' '+inference_hypothesis
    case2txt[casename]=case2txt[casename].strip(' ')


# build mapping of subsection key -> text
key2bounds={}
for filename in glob.glob(boundaries_dir+'/*'):
    for line in open(filename,'r'):
        start,stop,sec_id=line.strip('\n').split(',')
        key2bounds.setdefault(sec_id,set())
        key2bounds[sec_id].add((int(start),int(stop)))
sec2txt={}
for filename in glob.glob(statutes_dir+'/*'):
    sec_id=filename.split('/')[-1]
    sec2txt[sec_id]=''.join([line for line in open(filename, 'r')])
    sec2txt[sec_id]='S'+sec2txt[sec_id][1:] # replace non ascii with ascii
key2txt={'tax': 'There is hereby imposed on the taxable income of every individual for every taxable year, a tax determined in accordance with the following.'}
for key in key2bounds:
    spans=sorted(key2bounds[key])
    section='section'+key.split('(')[0]
    text=[]
    for s in spans:
        start,stop=s
        snippet=sec2txt[section][start:stop+1] # spans are inclusive
        _t=snippet.replace('\n',' ')
        text.append(_t)
    # normally should use a space below, but it messes with non-contiguous paragraphs
    joining=len(text)>1
    text=''.join(text)
    key2txt[key]=text

# build mapping of subsection key -> spans
section2spans={}
for filename in glob.glob(spans_dir+'/section*'):
    sec_id=filename.split('/')[-1]
    spans=filter(lambda x: x.strip('\n').strip(' ')[0]!='#',\
            open(filename,'r'))
    spans=map(lambda x: tuple(x.strip('\n').split(',')), spans)
    spans=map(lambda x: (int(x[0]), int(x[1]), x[2].upper()), spans)
    section2spans[sec_id]=set(spans)
key2spans={'tax': sorted([(27,44,'TAXINC'), (55,64,'TAXP'), \
        (76,87,'TAXY'), (90,94,'TAX')])}
for key in key2bounds:
    key2spans.setdefault(key,set())
    section='section'+key.split('(')[0]
    bounds=sorted(key2bounds[key])
    for span in sorted(section2spans[section]):
        for ii,b in enumerate(bounds):
            if ( (b[0]<=span[0]) and (span[1]<=b[1]) ):
                # means this span is relevant. Offset it
                offset=0
                for jj in range(ii):
                    offset-=bounds[jj][1]-bounds[jj][0]+1
                offset+=b[0]
                nu_span=(span[0]-offset, span[1]-offset, span[2])
                key2spans[key].add(nu_span)
    key2spans[key]=sorted(key2spans[key])
# split into train and test, and into gold and automatic
splits={'train': [line.strip('\n') for line in \
        open(splits_dir+'/train_split.txt','r')],\
        'test': [line.strip('\n') for line in \
        open(splits_dir+'/test_split.txt','r')]}
data={'train': {'gold':[], 'automatic': []}, \
        'test': {'gold': [], 'automatic': []}}
def upper_case_dict(d):
    # return same dict but with upper case keys
    output={}
    for k,v in d.items():
        output[k.upper()]=v
    return output
for key, filename in zip(['gold','automatic'],\
        [gold_slot_filling_annotations, \
        automatic_slot_filling_annotations]):
    for line in open(filename,'r'):
        try:
            assert '\t' in line
            case,predicate,in_slots,out_slots=line.strip('\n').split('\t')
            if out_slots=="false":
                out_slots={"@truth": False}
            else:
                out_slots=json.loads(out_slots)
                assert isinstance(out_slots,dict)
                for k in out_slots: # heuristic to make values more natural
                    if isinstance(out_slots[k],str):
                        out_slots[k]=out_slots[k].replace('_',' ')
                        out_slots[k]=out_slots[k].replace(' s ',"'s ")
                out_slots["@truth"]=True
            out_slots=upper_case_dict(out_slots)
            in_slots=json.loads(in_slots)
            in_slots=upper_case_dict(in_slots)
        except:
            print(line.strip('\n').split('\t'))
        if case.endswith('.pl'):
            case=case[:-3]
        if case in splits['train']:
            data['train'][key].append((case,predicate,in_slots,out_slots))
        elif case in splits['test']:
            if key=='gold': # otherwise, it can't be part of the test set
                data['test'][key].append((case,predicate,in_slots,out_slots))
                data['test']['automatic'].append((case,predicate,in_slots,out_slots))
        else:
            print('problem with ' + case)
            assert False
assert data['test']['gold'] == data['test']['automatic']
dependencies=[]
for filename in glob.glob(structure_dir+'/*'):
    stuff=map(lambda x: x.strip('\n').strip(' '), open(filename,'r'))
    stuff=' '.join(map(lambda x: x.strip(' '), stuff))
    stuff=stuff.split('.')
    stuff=map(lambda x: x.strip(' '), stuff)
    dependencies.extend(stuff)
while '' in dependencies:
    dependencies.remove('')
output={'train': data['train'], 'test': data['test'], \
        'statutes': key2txt, 'cases': case2txt, 'spans': key2spans,
        'structure': dependencies}
with open(output_file,'w') as f:
    json.dump(output,f,indent=2,sort_keys=True)
