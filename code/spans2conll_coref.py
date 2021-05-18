import sys, glob

parser = argparse.ArgumentParser(description='Convert spans to conll coref')
parser.add_argument('--spans', type=str, required=True,
                    help='path to statutes folder')
parser.add_argument('--boundaries', type=str, required=True,
                    help='path to boundaries folder')
parser.add_argument('--savefile', type=str, required=True,
                    help='name of file to write output to')
args = parser.parse_args()

OPEN_BRACKETS='[['
CLOSE_BRACKETS=']]'
SPAN_TEXT=['<COREF ID="','" TYPE="IDENT">','</COREF>']
BOUNDARIES_DIR=args.boundaries 
SPANS_DIR=args.spans

def filter_spans(boundaries,spans):
    output=[]
    for s in spans:
        start,end,label=s
        for b in boundaries:
            start_b,end_b=b
            in_b = (end>=start_b) and (start<=end_b) # ie there is some overlap
            if in_b: # only keep the part of the span that is within the boundaries
                new_start=max(start_b,start)
                new_end=min(end_b,end) 
                output.append((new_start,new_end,label))
                break
    return output

boundaries={}
output={}
# each subsection is treated as a distinct document
for filename in glob.glob(SPANS_DIR.rstrip('/')+'/*'):
    spans=open(filename,'r')
    spans=map(lambda x: x.strip('\n'), spans)
    spans=filter(lambda x: x.strip()[0]!='#', spans) # remove empty lines
    spans=filter(lambda x: len(x.strip('\n').strip(' '))>0, spans) # remove empty lines
    lookup={'start':{}, 'stop':{}} # make indexing of spans faster
    section_name=filename.split('/')[-1]
    output[section_name]=[]
    for s in spans:
        s=s.split(',')
        s=(int(s[0]),int(s[1]),s[2])
        start,stop,label=s
        assert stop>=start, str(s)
        output[section_name].append(s)
    # load section boundaries
    boundaries[section_name]={}
    for x in open(BOUNDARIES_DIR+'/'+section_name,'r'):
        start,end,name=x.strip('\n').split(',')
        boundaries[section_name].setdefault(name,set())
        boundaries[section_name][name].add((int(start),int(end)))
file_output=['#begin document (SARA);']
coref_index=[] # keep a running index of coref clusters
for section in sorted(boundaries.keys()):
    for subsection in sorted(boundaries[section].keys()):
        section_id=subsection.replace('(','_').replace(')','')
        filtered_spans=filter_spans(boundaries[section][subsection],output[section])
        for s in filtered_spans:
            # figure out whether it's a cluster
            span_id=section_id+'_'+s[2]
            count=sum(1 if x[2]==s[2] else 0 for x in filtered_spans)
            is_cluster=count>1
            file_output.append(section_id)
            if is_cluster:
                if span_id not in coref_index:
                    coref_index.append(span_id)
                index=coref_index.index(span_id)
                file_output[-1]+=' ('+str(index)+')'
            else:
                file_output[-1]+=' -'
        file_output.append('')
file_output.append('#end document')
with open(args.savefile, 'w') as f:
    f.write('\n'.join(file_output)+'\n')
