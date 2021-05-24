# do string matching-style coreference based on the output of the BERT-based BIO tagger
import sys, json, glob, copy
input_dir=sys.argv[1]
output_file=sys.argv[2]

def normalize_string(s):
    nullwords=['such','a','an','the','any','his','every']
    x=str(s).split(' ')
    for w in nullwords:
        if w in x:
            x.remove(w)
    x=' '.join(x).strip().lower()
    x=x.replace('  ',' ')
    return x

def similarity(surface1,surface2): # deciding whether two spans belong to the same cluster or not
    x1=normalize_string(surface1)
    x2=normalize_string(surface2)
    return 1 if x1==x2 else 0

f=open(output_file,'w')
for split in range(7):
    filename=input_dir.rstrip('/')+'/split_'+str(split+1)+'/test_outputs.json'
    for line in open(filename,'r'):
        # one example per line
        table={}
        data=json.loads(line.strip('\n'))
        for span in data['predicted_spans']:
            start,stop=span
            surface_form=' '.join(data['tokens'][start:stop+1])
            surface_form=surface_form.replace(' ##','')
            x=normalize_string(surface_form)
            table.setdefault(x,set())
            table[x].add(tuple(span))
        clusters=[]
        for x in table:
            clusters.append(sorted(list(y) for y in table[x]))
        data['predicted_clusters']=clusters
        f.write(json.dumps(data)+'\n')
f.close()
