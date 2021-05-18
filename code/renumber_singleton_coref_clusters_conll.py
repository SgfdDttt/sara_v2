import sys
input_file=sys.argv[1]
output_file=sys.argv[2]
counter=0
output=[]
for line in open(input_file,'r'):
    line=line.strip('\n')
    if '#' in line:
        continue
    if '(' in line:
        c=int(line.split('(')[1].split(')')[0])
        counter=max(c,counter)
counter+=1
for line in open(input_file,'r'):
    line=line.strip('\n')
    if '-' in line:
        line=line.replace('-','('+str(counter)+')')
        counter+=1
    output.append(line)
with open(output_file, 'w') as f:
    f.write('\n'.join(output)+'\n')
