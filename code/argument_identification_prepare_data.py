import argparse, glob, json
from transformers import BertTokenizer


parser = argparse.ArgumentParser(description='Prepare argument identification data for BIO tagger')
parser.add_argument('--statutes', type=str,
                    help='path to statutes folder')
parser.add_argument('--spans', type=str,
                    help='path to span folder')
parser.add_argument('--boundaries', type=str,
                    help='path to boundaries folder')
parser.add_argument('--savefile', type=str,
                    help='name of file to save data in')
parser.add_argument('--max_length', type=int, default=512,
                    help='max length of input string')
parser.add_argument('--bert_tokenizer', type=str, default='bert-base-cased',
                    help='which bert tokenizer to use')
args = parser.parse_args()

# load data
STATUTES={}
for filename in glob.glob(args.statutes+'/*'):
    statute_id=filename.split('/')[-1] 
    STATUTES[statute_id]=''.join(open(filename,'r'))
BOUNDARIES={}
for filename in glob.glob(args.boundaries+'/*'):
    statute_id=filename.split('/')[-1] 
    BOUNDARIES[statute_id]=set()
    for line in open(filename,'r'):
        if line.strip(' ')[0]=='#':
            continue
        start,stop,subsec_id=line.strip('\n').strip(' ').split(',')
        start=int(start)
        stop=int(stop)
        BOUNDARIES[statute_id].add((start,stop,subsec_id))

def get_subsec_id(section_id,start,end):
    # based on start and end of span, find corresponding subsection
    # some matches are off by a bit so take best match
    section_id = section_id if section_id.startswith('section') else 'section'+section_id
    assert start<=end
    best_match,best_offset=None,None
    for ii,jj,sid in BOUNDARIES[section_id]:
        offset=abs(ii-start)+abs(jj-end)
        if best_match is None:
            best_offset=offset
            best_match=sid
        else:
            if offset<best_offset:
                best_match=sid
                best_offset=offset
    output=best_match
    assert output is not None
    return output

# character level BIO tags
TAGS={}
for filename in glob.glob(args.spans+'/*'):
    statute_id=filename.split('/')[-1] 
    TAGS[statute_id]=[('O',None) for _ in STATUTES[statute_id]]
    assert len(TAGS[statute_id])==len(STATUTES[statute_id])
    for line in open(filename,'r'):
        if line[0]=='#':
            continue
        start,stop,name=line.strip('\n').split(',')
        start=int(start); stop=int(stop)
        TAGS[statute_id][start]=('B',name)
        for ii in range(start+1,stop+1):
            TAGS[statute_id][ii]=('I',name)
# quick check for correctness
for k,v in TAGS.items():
    for a,b in zip(v[:-1],v[1:]):
        assert not ((a=='O') and (b=='I'))

DATA={'input': [], 'target': [], 'spans': [], 'section': [], 'subsection': []}
for key in STATUTES:
    section_id=key[len('section'):]
    # cut up into paragraphs
    start,end=0,0
    while start<len(STATUTES[key]):
        end=start+1
        while STATUTES[key][end]!='\n':
            end+=1
            if end==len(STATUTES[key]):
                break
        chunk=STATUTES[key][start:end]
        subsection_id=get_subsec_id(section_id,start,end)
        assert '\n' not in chunk
        tags=TAGS[key][start:end]
        spans=set()
        start_span,end_span=0,0
        while start_span<len(tags):
            if tags[start_span][0]=='B':
                end_span=start_span+1
                while tags[end_span][0]=='I':
                    end_span+=1
                    if end_span==len(tags):
                        break
                if (tags[start_span][0]=='B') and (tags[end_span][0] in ['B','O']):
                    name=tags[start_span][1]
                    spans.add((start_span,end_span-1,name))
                start_span=end_span-1
            start_span+=1
        # end while start_span<len(tags)
        DATA['input'].append(chunk)
        DATA['target'].append([x[0] for x in tags])
        DATA['spans'].append(sorted(spans))
        DATA['section'].append(section_id)
        DATA['subsection'].append(subsection_id)
        start=end
        if start==len(STATUTES[key]):
            break
        while STATUTES[key][start]=='\n':
            start+=1
            if start==len(STATUTES[key]):
                break
    # end while end<len(self.statutes[key])
# end for key in keys

# tokenize etc
def tokenize(x):
    tokens=TOKENIZER.tokenize(x)
    # create alignment map source char index -> target token index
    src_ind,tgt_ind=0,0
    buff=""
    src=x
    tgt=tokens
    # the mapping goes from src to tgt, int -> int
    # src is a str (list of characters) while tgt is a list (of tokens)
    mapping=[ None for _ in src ]
    while (src_ind<len(src)) and (tgt_ind<len(tgt)):
        if src[src_ind]==' ': # BERT tokenizer removes spaces
            src_ind+=1; continue
        mapping[src_ind]=tgt_ind
        if tgt[tgt_ind]=='[UNK]':
            buff=""
            while src[src_ind] not in [' ',',',':','.']:
                mapping[src_ind]=tgt_ind
                src_ind+=1
                if src_ind==len(src):
                    break
            tgt_ind+=1
            continue
        buff+=src[src_ind]
        if buff==tgt[tgt_ind].lstrip('##'):
            buff=""
            tgt_ind+=1
        src_ind+=1
    # check that mapping is valid
    tgt_side=sorted(set(filter(lambda x: x is not None, mapping)))
    if tgt_side!=list(range(len(tgt))):
        print('problem with ' + str(tgt_side) \
                + ' vs ' + str(len(tgt)))
    return tokens,mapping
# end def tokenize

TOKENIZER=BertTokenizer.from_pretrained(args.bert_tokenizer)
DATA['tokens']=[]
DATA['mapping']=[]
DATA['projected_targets']=[]
DATA['projected_spans']=[]
DATA['bert_input']=[]
for INP,TARG,SPANS in zip(DATA['input'],DATA['target'],DATA['spans']):
    assert len(INP)==len(TARG)
    # tokenize
    tokens,mapping=tokenize(INP)
    assert len(tokens)<=args.max_length
    DATA['tokens'].append(tokens)
    DATA['mapping'].append(mapping)
    # argument spans
    nu_spans=set()
    for span in SPANS:
        start,stop,name=span
        nu_start=mapping[start]
        nu_stop=mapping[stop]
        if nu_start is None:
            nu_start=mapping[start+1]
        if nu_stop is None:
            nu_stop=mapping[stop-1]
        assert nu_start is not None
        assert nu_stop is not None
        nu_spans.add((nu_start,nu_stop,name))
    assert len(nu_spans)==len(SPANS)
    DATA['projected_spans'].append(sorted(nu_spans))
    # adjust target
    target=[set() for _ in tokens]
    for jj,_l in enumerate(TARG):
        tok_index=mapping[jj]
        if tok_index is not None:
            target[tok_index].add(_l)
    # end for jj,_l in enumerate(_y)
    for ii,s in enumerate(target):
        if len(s)==0:
            target[ii]='' # no label because it's a space
        elif len(s)==1:
            target[ii]=list(s)[0]
        elif 'B' in s:
            target[ii]='B'
        else:
            target[ii]='I'
    # end for ii,s in enumerate(target)
    while '' in target:
        target.remove('')
    assert all(isinstance(x,str) for x in target)
    DATA['projected_targets'].append(target)
    # feed through bert
    input_ids=TOKENIZER.encode(tokens)#,is_pretokenized=True)
    tset=TOKENIZER.decode(input_ids)
    assert len(input_ids)==len(tokens)+2, str(input_ids) + ' vs ' + str(tokens)
    token_type_ids = [0]*len(input_ids)
    input_mask = [1]*len(input_ids)
    DATA['bert_input'].append(input_ids)
DATA['projected_clusters']=[ [] for _ in DATA['projected_spans'] ]
for ii,spans in enumerate(DATA['projected_spans']):
    cluster_index={}
    for start,stop,name in spans:
        cluster_index.setdefault(name,len(cluster_index))
        ind=cluster_index[name]
        if len(DATA['projected_clusters'][ii])==ind:
            DATA['projected_clusters'][ii].append([])
        DATA['projected_clusters'][ii][ind].append([start,stop])
    DATA['projected_spans'][ii]=[ (x[0],x[1]) for x in DATA['projected_spans'][ii] ]
for k in DATA:
    assert len(DATA[k])==len(DATA['input'])
for ii,_ in enumerate(DATA['input']):
    assert len(DATA['input'][ii])==len(DATA['target'][ii])
    assert len(DATA['bert_input'][ii])==len(DATA['tokens'][ii])+2, str(DATA['tokens'][ii]) + ' vs ' + str(DATA['bert_input'][ii])
    assert len(DATA['tokens'][ii])==len(DATA['projected_targets'][ii])
    assert len(DATA['bert_input'][ii])==len(DATA['projected_targets'][ii])+2

json.dump(DATA,open(args.savefile,'w'),indent=2,sort_keys=True)
