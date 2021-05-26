# simpler baseline: each mention is its own cluster
import glob, os, argparse

parser = argparse.ArgumentParser(description='Run single mention coref baseline')
parser.add_argument('--statutes', type=str, required=True,
                    help='path to statutes folder')
parser.add_argument('--spans', type=str, required=True,
                    help='path to span folder')
parser.add_argument('--output_dir', type=str, required=True,
                    help='name of file to save data in')
args = parser.parse_args()

span_dir=args.spans.rstrip('/') # for each section, list of (start_offset,end_offset,*), offsets refer to sections
statute_dir=args.statutes.rstrip('/') # text of the statutes
output_dir=args.output_dir.rstrip('/') # where to write the results to
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

for section_file in sorted(glob.glob(statute_dir+'/*'))[::-1]:
    cluster_counter=0
    output=[]
    section_name=section_file.split('/')[-1]
    sec2spans={}
    for line in open(span_dir+'/'+section_name,'r'):
        content=line.strip('\n').strip()
        if content[0]=='#':
            current_section=content.strip('# ')
            sec2spans.setdefault(current_section,[])
        else:
            start,stop,_=content.split(',')
            sec2spans[current_section].append((int(start),int(stop)))
    for section in sec2spans:
        sec2spans[section].sort()
    text=''.join(open(section_file,'r'))
    sec2clusters={}
    for section,spans in sec2spans.items():
        # create clusters
        sec2clusters.setdefault(section,{}) # map spans to their cluster
        for s1 in spans:
            cluster='C'+str(cluster_counter)
            cluster_counter+=1
            sec2clusters[section][s1]=cluster
    with open(output_dir+'/'+section_name,'w') as f:
        for section in sorted(sec2clusters.keys()):
            f.write('# ' + section + '\n')
            for span in sorted(sec2clusters[section].keys()):
                f.write(','.join(str(x) for x in span) + ',' + sec2clusters[section][span] + '\n')
