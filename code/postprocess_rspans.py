import glob, os, argparse

parser = argparse.ArgumentParser(description='Postprocess rspans')
parser.add_argument('--statutes', type=str, required=True,
                    help='path to statutes folder')
parser.add_argument('--input', type=str, required=True
                    help='path to rspans')
parser.add_argument('--output', type=str, required=True,
                    help='name of folder to write output to')
args = parser.parse_args()

statutes_dir=args.statutes.rstrip('/')
input_file=args.input
output_dir=args.output.rstrip('/')
try:
    os.mkdir(output_dir)
except:
    pass
# load spans and text
spans,text=[],[]
for ii,line in enumerate(open(input_file,'r')):
    if ii%2==0:
        text.append(line.strip('\n'))
    else:
        spans.append(line.strip('\n').split(' '))
# convert to full-text statutes
alt_statutes,alt_spans={},{}
for t,s in zip(text,spans):
    if '§' in t:
        current_section=t.split('§')[1].strip(' ').split(' ')[0].strip('. ')
        assert current_section not in alt_statutes
        alt_statutes[current_section]=[]
        assert current_section not in alt_spans
        alt_spans[current_section]=[]
    alt_statutes[current_section].append(t)
    alt_spans[current_section].append(s)
# load original statutes
og_statutes={}
for filename in glob.glob(statutes_dir+'/*'):
    statute_id=filename.split('/')[-1][7:]
    assert statute_id not in og_statutes
    og_statutes[statute_id]='\n'.join([line.strip('\n') for line in open(filename,'r')])
# build char maps from alt to original
char_maps={}
for section in alt_statutes:
    char_counter=0
    assert section not in char_maps
    char_maps[section]={}
    for line_num,line in enumerate(alt_statutes[section]):
        for char_num,char in enumerate(line):
            if char in [' ','\n']:
                continue # these will never be span boundaries anyway
            while og_statutes[section][char_counter]!=char:
                char_counter+=1
            assert (line_num,char_num) not in char_maps[section]
            char_maps[section][(line_num,char_num)]=char_counter
            char_counter+=1 # avoid staying in place for repeated chars
    # sanity check
    for k,v in char_maps[section].items():
        assert og_statutes[section][v]==alt_statutes[section][k[0]][k[1]]
for section in alt_spans:
    output=[]
    for line_num,span_list in enumerate(alt_spans[section]):
        for span in span_list:
            old_start,old_end=span.split('-')[:2]
            new_start=char_maps[section][(line_num,int(old_start))]
            new_end=char_maps[section][(line_num,int(old_end))]
            output.append((new_start,new_end))
    output.sort()
    with open(output_dir+'/section'+section,'w') as f:
        counter=0
        for x in output:
            f.write(str(x[0])+','+str(x[1])+',S'+str(counter)+'\n')
            counter+=1
