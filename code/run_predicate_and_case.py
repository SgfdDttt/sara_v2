from pyswip import Prolog, Variable
import glob,re,os,json,random,copy,itertools,argparse

parser = argparse.ArgumentParser(description='Run Prolog code on all predicate and case pairs')
parser.add_argument('--prolog', type=str, required=True,
                    help='path to folder containing Prolog program')
parser.add_argument('--cases', type=str, required=True,
                    help='path to folder containing cases')
parser.add_argument('--case_id', type=int, required=True,
                    help='index of case to consider')
parser.add_argument('--predicate_id', type=int, required=True,
                    help='index of predicate to consider')
parser.add_argument('--tmp_file', type=str, required=True,
                    help='string to use for a tmp file')
args = parser.parse_args()


CASE_DIR=args.cases.rstrip('/') # where all the cases are
PREDICATE_DIR=args.prolog.rstrip('/') # where all the prolog predicates are
YEAR_REGEXP=re.compile(r"[^$]?20[0-3][0-9]|19[0-9]{2}|18[0-9]{2}")
DATE_REGEXP=re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
PERSON_ARGUMENTS=sorted(['Bsssc2b', 'Childc2a', 'Dependent', 'Employee', 'Employer', 'Individual', 'Individuals', 'Otaxp', 'Payee', 'Payer', 'Person', 'Person_list', 'S113', 'S121', 'S125', 'S24', 'S45', 'S65', 'Spouse', 'Student', 'Surviving_spouse', 'Taxp'])

COORDINATES=(args.case_id, args.predicate_id)
SPECIAL_ID=args.tmp_file # a special id for the tmp file
random.seed(sum(COORDINATES))

class Predicate:
    def __init__(self,string):
        self.name=string.split('(')[0]
        assert self.name.startswith('s') or self.name=='tax', string
        self.arguments=string[len(self.name):]
        self.arguments=self.arguments.strip('()').split(',')
        self.arguments=[x.strip(' ') for x in self.arguments]

    def __lt__(self,other): # for sorting purposes
        return self.name<other.name

    def to_string(self,substitutions={}):
        args=[]
        for x in self.arguments:
            if x in substitutions:
                arg=substitutions[x]
            else:
                arg=x
            args.append(arg)
        args=[str(x) for x in args]
        return self.name+'('+','.join(args)+')'

class Case:
    TEXT_KEY='% Text'
    QUESTION_KEY='% Question'
    FACTS_KEY='% Facts'
    QUERY_KEY='% Test'
    def __init__(self,filename,string):
        self.filename=filename
        lines=string.split('\n')
        text_index=lines.index(Case.TEXT_KEY)
        question_index=lines.index(Case.QUESTION_KEY)
        facts_index=lines.index(Case.FACTS_KEY)
        query_index=lines.index(Case.QUERY_KEY)
        self.text=[x.strip('% ') for x in lines[text_index+1:facts_index]]
        self.text=list(filter(lambda x: len(x)>0, self.text))
        self.question=lines[question_index+1]
        self.facts=lines[facts_index+1:query_index-1]
        self.facts=map(lambda x: x.strip('\n').strip(' '), self.facts)
        self.facts=''.join(self.facts)
        split_points=[]
        in_string=False
        for ii,c in enumerate(self.facts):
            if c=='"':
                in_string = not in_string
            elif c in ['.','%']:
                if in_string:
                    continue
                else:
                    split_points.append(ii)
        split_points.insert(0,0)
        split_points.append(len(self.facts))
        new_facts=[]
        for start,stop in zip(split_points[:-1],split_points[1:]):
            new_facts.append(self.facts[start:stop])
        self.facts=new_facts
        self.facts=map(lambda x: x.strip('.'), self.facts)
        self.facts=list(filter(lambda x: len(x)>0, self.facts))
        assert '' not in self.facts
        self.query=''.join(lines[query_index+1:]).split('.')[0]
        self.query=self.query.strip('.:- ') 

    def __lt__(self,other): # for sorting purposes
        return self.filename<other.filename

class PrologEnv:
    def __init__(self,predicates):
        self.engine=Prolog()
        # make all law predicates dynamic
        tmp_file=PREDICATE_DIR+'/tmp'+SPECIAL_ID+'.pl'
        # put in all predicates
        with open(tmp_file,'w') as f:
            for p in predicates:
                assertion=':- dynamic ' + p.name + '/' + str(len(p.arguments)) + '.'
                f.write(assertion+'\n')
            f.write(':- dynamic s151_b_applies/2.\n')
            f.write(':- dynamic total_wages_employer/6.\n')
            # add all events
            f.write(''.join(open(PREDICATE_DIR+'/events.pl','r')))
            # all utils
            f.write(''.join(open(PREDICATE_DIR+'/utils.pl','r')))
            # each law file
            for thing in sorted(glob.glob(PREDICATE_DIR+'/section*.pl')):
                f.write(''.join(open(thing,'r')))
        # consult file
        self.engine.consult(tmp_file)
        # remove file
        try:
            os.remove(tmp_file)
        except:
            pass

    def assertz(self,input):
        self.engine.assertz(input)

    def query(self,input):
        return self.engine.query(input)

def dict_to_string(d):
    print('dict_to_string')
    print(d)
    output=json.dumps(d)
    return output

def is_variable(s):
    if not isinstance(s,str):
        return False
    return (s[0]=='_') or (s.startswith('Variable('))

def format_query_result(d):
    output={}
    for k in sorted(d.keys()):
        v=d[k]
        print(v,type(v),isinstance(v,Variable))
        if isinstance(v,list):
            output[k]=[]
            for x in v:
                if isinstance(x,bytes):
                    x=x.decode('utf-8')
                if not is_variable(x):
                    output[k].append(x)
        elif not isinstance(v,Variable):
            if not is_variable(v):
                output[k]=v
    return output

all_predicates=[]
for thing in sorted(glob.glob(PREDICATE_DIR+'/section*.pl')):
    lines=[line.strip('\n').split('%')[0] for line in open(thing,'r')]
    predicates=map(lambda x: x.replace(':-','').strip(' '),
            filter(lambda x: (x.startswith('s')), lines))
    all_predicates.extend([Predicate(x) for x in predicates])
lines=[line.strip('\n').split('%')[0] for line in open(PREDICATE_DIR+'/utils.pl','r')]
predicates=list(map(lambda x: x.replace(':-','').strip(' '),
	filter(lambda x: (x.startswith('tax')), lines)))
assert len(predicates)==1
all_predicates.append(Predicate(predicates[0]))

all_cases=[]
for thing in sorted(glob.glob(CASE_DIR+'/*')):
    all_cases.append(Case(thing.split('/')[-1],''.join(open(thing,'r'))))

case=sorted(all_cases)[COORDINATES[0]]
years=[int(x) for x in YEAR_REGEXP.findall(case.question)]
years=list(filter(lambda x: (x>1800) and (x<2200), years))
potential_characters=['Alice', 'Bob', 'Cameron', 'Charlie', 'Dan', \
	'Dorothy', 'Emily', 'Fred', 'George', 'Harold']
characters_present=list(filter(lambda x: x in ' '.join(case.text), potential_characters))+[None]
predicate=sorted(all_predicates)[COORDINATES[1]]
substitution_sets=list()
person_arguments=list(filter(lambda x: x in PERSON_ARGUMENTS, predicate.arguments))
character_indices=[ 0 for _ in person_arguments ]
if len(person_arguments)>0:
    while character_indices[-1]<len(characters_present):
        substitution={}
        for p,c in zip(person_arguments,character_indices):
            if characters_present[c] is not None:
                substitution[p]='"'+characters_present[c]+'"'
        substitution_sets.append(dict(substitution))
        character_indices[0]+=1
        for jj in range(len(character_indices)-1):
            if character_indices[jj]==len(characters_present):
                character_indices[jj]=0
                character_indices[jj+1]+=1
else:
    substitution_sets.append({})
year_key='Taxy'
if (year_key not in predicate.arguments) and ('Caly' in predicate.arguments):
    year_key='Caly'
person_sets=list(substitution_sets)
substitution_sets=[]
for s in person_sets:
    for year in years:
        t=dict(s)
        t[year_key]=str(year)
        substitution_sets.append(t)
    else:
        substitution_sets.append(dict(s))
substitution_sets.extend(person_sets)
if {} not in substitution_sets:
    substitution_sets.append({})
prolog=PrologEnv(all_predicates)
for fact in case.facts:
    if fact.startswith(':-'):
        continue
    prolog.assertz(fact)
# first, check that the original test passes (sanity check)
results=list(prolog.query(case.query))
assert len(results)>0

def run_query(query):
    try:
        results=list(prolog.query(query))
    except:
        results=[]
    results=[format_query_result(x) for x in results]
    return results

def has_unbound_variable(result):
    has_unbound=False
    for _,v in result.items():
        if isinstance(v,list):
            has_unbound |= any(x is None for x in v)
        else:
            has_unbound |= v is None
    return has_unbound

def values_not_in_text(result):
    invalid=False
    case_text=' '.join(case.text)
    for _,v in result.items():
        w=v if isinstance(v,list) else [v]
        for x in w:
            is_year=bool(YEAR_REGEXP.match(str(x)))
            if isinstance(x,str):
                is_date=bool(YEAR_REGEXP.match(str(x)))
                if is_date: # those will never be in text
                    continue
                else:
                    invalid |= x not in case_text
            elif is_year: # years ought to be in text
                invalid |= str(x) not in case_text
            else:
                continue # because then it's an integer and those aren't in text
    return invalid

def corrupt_query(query_input,include_output=False,use_all_characters=False):
    # sample new values for each slot until the query fails
    query=copy.deepcopy(query_input)
    characters=list(filter(lambda x: x is not None, characters_present))
    if use_all_characters:
        characters=potential_characters
    characters=['"'+x+'"' for x in characters]
    random.shuffle(characters)
    all_corruptions=[]
    if include_output:
        results=run_query(predicate.to_string(query))
        assert len(results)>0, "input is supposed to be a positive query"
        query.update(results[0])
    for key in sorted(query.keys()):
        possible_values=[query[key]]
        if key in PERSON_ARGUMENTS:
            possible_values=characters
        elif key==year_key:
            possible_values=[str(x) for x in years]
        else:
            try:
                # pick 10 values such that the numerical accuracy is 1
                value=int(query[key])
                denom=int(max(0.1*value,5000))
                # +1 and -1 for strict inequality
                lower_bound=max(0,value-denom)+1
                upper_bound=max(0,value+denom)-1
                a=(upper_bound-lower_bound)/9.0
                possible_values=sorted(set(
                    str(int(a*ii+lower_bound)) for ii in range(10) ))
            except:
                pass
        all_corruptions.append(possible_values)
    all_corruptions=itertools.product(*all_corruptions)
    negative_queries=list()
    for corruption in all_corruptions:
        substitution={}
        for ii,key in enumerate(sorted(query.keys())):
            substitution[key]=str(corruption[ii])
        result=run_query(predicate.to_string(substitution))
        if len(result)==0:
            if substitution not in negative_queries:
                negative_queries.append(substitution)
    return negative_queries

# run all the generated queries
queries_done_already=set()
negative_queries={'A': [], 'B': [], 'C': [], 'D': []} # 4 tiers of negative queries
positive_queries=set()
for substitution in substitution_sets:
    query_str=predicate.to_string(substitution)
    if query_str in queries_done_already:
        continue
    results=run_query(query_str)
    queries_done_already.add(query_str)
    if len(results)==0:
        if substitution not in negative_queries['C']:
            negative_queries['C'].append(substitution) # added in tier C since it's a "random" negative query
        continue
    results=list(filter(lambda x: not (has_unbound_variable(x) or values_not_in_text(x)), results))
    if len(results)==0:
        continue
    for rr in results:
        positive_queries.add(str(case.filename)+'\t'+str(predicate.name) \
                +'\t'+dict_to_string(format_query_result(substitution)) \
                +'\t'+dict_to_string(rr))
    # Tier A queries: some arguments left open
    negative_queries['A'].extend(corrupt_query(substitution)) 
    # Tier B queries: all arguments specified
    negative_queries['B'].extend(corrupt_query(substitution,True))
    # Tier D is pretty much trash, since it involves characters not even
    # present in the case
    negative_queries['D'].extend(corrupt_query(substitution,True,True))
# take A-tier queries, then B-tier, then C-tier
random.shuffle(negative_queries['A'])
random.shuffle(negative_queries['B'])
random.shuffle(negative_queries['C'])
random.shuffle(negative_queries['D'])
positive_queries=sorted(positive_queries)
random.shuffle(positive_queries)
while (len(positive_queries)>0):
    substitution=positive_queries.pop(0)
    print(substitution)
for key in ['A','B','C','D']:
    nq=negative_queries[key]
    while (len(nq)>0):
        substitution=nq.pop(0) # take queries from the top of the stack
        print(key+'\t'+str(case.filename)+'\t'+str(predicate.name) \
            +'\t'+dict_to_string(format_query_result(substitution)) \
            +'\tfalse')
