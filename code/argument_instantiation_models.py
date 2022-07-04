import logging, string, json, math, random, re, datetime, copy
import torch
from torch import nn
import numpy as np
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from transformers import *

TRUTH_KEY='@TRUTH'
DATE_REGEXP=re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
DATE_FORMATS=[ # ordered by priority
        '%b %-d, %Y', # Jan 1, 2021
        '%B %-d, %Y', # January 1, 2021
        '%b %-dst, %Y', # Feb 1st, 2017
        '%B %-dst, %Y', # October 1st, 2013
        '%b %-dnd, %Y', # Mar 2nd, 2015
        '%B %-dnd, %Y', # March 2nd, 2015
        '%b %-drd, %Y', # Feb 3rd, 1992
        '%B %-drd, %Y', # May 29th, 2008
        '%b %-dth, %Y',
        '%B %-dth, %Y', # July 9th, 2014

        '%b %-d', # Jan 1
        '%B %-d', # January 1
        '%b %-dst', # Feb 1st
        '%B %-dst', # October 1st
        '%b %-dnd', # Mar 2nd
        '%B %-dnd', # March 2nd
        '%b %-drd', # Feb 3rd
        '%B %-drd', # May 29th
        '%b %-dth',
        '%B %-dth', # July 9th

        '%Y' # 2017
        ]

KEYS=[
        ['a','b','c','d','e','f','g'],
        [str(ii) for ii in range(1,22)],
        ['A','B','C','D','E','F','G','H'],
        ['i','ii','iii','iv','v'],
        ['I','II']
    ]


def get_argument_type_arg_only(arg_name):
    ''' this is somewhat like cheating, it
    tells you how to predict this argument
    (binary, integer, or span) '''
    arg_types={
            'integer': ["WAGES","TAXINC","TAX","STANDED","S96","S9A","S9B","S7","S48","COUNTS",
                "S33A","S30B","S27","S156","S14","S13A","S2","GROSSINC","EA","DED63","DED151",
                "COST","BASSD","AP","AGI","ADDSD","AA","ADDITIONAL_AMOUNTS","AMOUNT",
                "EXEMPTIONS_LIST","MULTIPLIER","TOTAL_AMOUNT","AMOUNT_DEDUCTIONS_OUT",
                "AMOUNT_DEDUCTIONS_IN","ITEMDED","REMUNERATION2"],
            'span': ["REASON", "WORKDAY","WIFE","TAXY","TAXP","STUDENT","SPOUSE","SERVICE",
                "S98","S92",
                "S9C","S89","S81","S8","S77","S65","S62","S61","S6","S59B","S59A","S5","S47",
                "S46B","S46A","S45","S44A","S41","S40A","S40B","S4","S38","S36","S34","S33B",
                "S32","S31","S30A","S24A","S24B","S24","S235","S23","S228","S227","S22","S21","S201",
                "S19","S16","S158","S145","S143","S138","S13B","S13","S125","S121","S119", "S119A","S119B",
                "S113","S112","S110","S105","S104","S101","S10","RELATIONSHIP","PRECCALY",
                "PLAN","PAYEE","OTAXP","MEDIUM","LOCATION","HUSBAND","HOUSEHOLD","FOM",
                "EMPLOYMENT","EMPLOYER","EMPLOYEE","DEPENDENT","CYB1","CHILDC2A","CALY",
                "BSSSC2B","S44B","DEDUCTION","PAYER","JOINT_RETURN","START_RELATIONSHIP",
                "END_RELATIONSHIP","MARRIAGE","END_DAY","START_DAY","SURVIVING_SPOUSE",
                "PERSON_LIST","REMUNERATION","PREVIOUS_MARRIAGE","REMARRIAGE"],
            'binary': ["@TRUTH"]
            }
    for k,v in arg_types.items():
        if arg_name in v:
            return k
    assert False, "every argument needs a type ("+arg_name+")"

def get_argument_type(arg_name,arg_value=None):
    arg_type=get_argument_type_arg_only(arg_name)
    if arg_value is None:
        return arg_type
    if arg_type=='binary':
        return arg_type
    if arg_type=='integer':
        try:
            val=int(arg_value)
            return arg_type
        except:
            return 'span'
    # don't try to cast a span to an integer because it could be a year
    return arg_type



class Metrics:
    def __init__(self):
        self.targets,self.predictions={},{}
        # 'binary': just the '@truth' values of binary cases
        # 'numerical': just the 'tax' values of numerical cases
        for key in ['all', 'binary', 'numerical']:
            if key=='all':
                self.targets[key]={'@truth':[], 'dollar_amounts':[], 'other_in':[],
                        'other_out': []}
                self.predictions[key]={'@truth':[], 'dollar_amounts':[], 'other_in':[],
                        'other_out': []}
            else:
                self.targets[key]=[]
                self.predictions[key]=[]

    def is_dollar_amount(self,arg_name,arg_value):
        # set of heuristics to decide whether something is a dollar amount
        for keyword in ['TAXY','CALY','YEAR']:
            if keyword in arg_name:
                return False
        for keyword in ['EA','TAX','AA','AGI','TAXINC','GROSSINC','WAGES']:
            if keyword==arg_name:
                return True
        try:
            arg_value=int(arg_value)
        except:
            return False
        if (arg_value<1900) or (arg_value>2100):
            return True
        return False

    def check_saved_data(self):
        is_ok=True
        for key in ['numerical','binary']:
            is_ok&=(len(self.targets[key])==len(self.predictions[key]))
        for key in ['@truth', 'dollar_amounts', 'other_in', 'other_out']:
            is_ok&=(len(self.targets['all'][key])==len(self.predictions['all'][key]))
        return is_ok

    def accumulate(self,batch):
        for predictions,queries,case_ref,case_text in zip(batch['predictions'],
                batch['query_slots'],batch['case_ref'],batch['context']):
            is_numerical_case=case_ref.startswith('tax_case_')
            for arg_name,arg_value in queries:
                # if absent from prediction, will be counted as false
                prediction=predictions.get(arg_name)
                argument_type=get_argument_type(arg_name,arg_value)
                assert argument_type in ['binary', 'integer', 'span']
                if argument_type=='binary':
                    # register for general metrics
                    self.targets['all']['@truth'].append(arg_value)
                    self.predictions['all']['@truth'].append(prediction)
                    if not is_numerical_case:
                        self.targets['binary'].append(arg_value)
                        self.predictions['binary'].append(prediction)
                elif argument_type=='integer':
                    arg_value_int=int(arg_value)
                    try:
                        prediction_int=int(prediction)
                    except:
                        prediction_int=None
                    self.targets['all']['dollar_amounts'].append(arg_value_int)
                    self.predictions['all']['dollar_amounts'].append(prediction_int)
                    if is_numerical_case and (arg_name=='TAX'):
                        self.targets['numerical'].append(arg_value_int)
                        self.predictions['numerical'].append(prediction_int)
                elif argument_type=='span':
                    key='other_in' if str(arg_value).lower() in case_text.lower()\
                            else 'other_out'
                    prediction_str=None if prediction is None else str(prediction)
                    logging.debug(' // '.join((str(arg_name),str(arg_value),str(prediction))))
                    self.targets['all'][key].append(str(arg_value))
                    self.predictions['all'][key].append(prediction_str)
            # end for arg_name,arg_value in queries
        # end for predictions,queries,case_ref in zip(batch['predictions'],
        # check that there is no length discrepancy
        assert self.check_saved_data()

    def mean(self,samples):
        return float(sum(samples))/max(1,len(samples))

    def binary_accuracy(self,predictions,targets):
        # based on exact match
        assert len(predictions)==len(targets)
        samples=[]
        for p,t in zip(predictions,targets):
            if p is not None:
                samples.append(1 if p==t else 0)
            else:
                samples.append(0)
        assert len(samples)==len(targets)
        return samples

    def relative_error(self,prediction,target):
        assert isinstance(prediction,int)
        assert isinstance(target,int)
        num=abs(prediction-target)
        denom=max(0.1*target,5000)
        return num/float(denom)

    def numerical_accuracy(self,predictions,targets):
        # numerical accuracy
        assert len(predictions)==len(targets)
        samples=[]
        for p,t in zip(predictions,targets):
            if p is not None:
                samples.append(1 if self.relative_error(p,t)<1 else 0)
            else:
                samples.append(0)
        assert len(samples)==len(targets)
        return samples

    def unigram_jaccard(self,predictions,targets):
        assert len(predictions)==len(targets)
        samples=[]
        for p,t in zip(predictions,targets):
            if p is not None:
                a, b = set(str(p).split(' ')), set(str(t).split(' '))
                num, denom = len(a & b), len(a | b)
                samples.append(float(num)/max(denom,1))
            else:
                samples.append(0)
        assert len(samples)==len(targets)
        return samples

    def core(self,compute_confidence_intervals=False):
        output={}
        output['binary accuracy']=self.binary_accuracy(
                self.predictions['binary'],
                self.targets['binary'])
        output['numerical accuracy']=self.numerical_accuracy(
                self.predictions['numerical'],
                self.targets['numerical'])
        output['core unified']=output['binary accuracy']+output['numerical accuracy']
        if compute_confidence_intervals:
            output['binary accuracy confidence interval']=\
                self.ninety_percent_confidence_interval(\
                output['binary accuracy'])
            output['numerical accuracy confidence interval']=\
                self.ninety_percent_confidence_interval(\
                output['numerical accuracy'])
            output['core unified confidence interval']=\
                    self.ninety_percent_confidence_interval(output['core unified'])
        output['binary accuracy']=self.mean(output['binary accuracy'])
        output['numerical accuracy']=self.mean(output['numerical accuracy'])
        output['core unified'] = self.mean(output['core unified'])
        return output

    def info(self,compute_confidence_intervals=False):
        output={}
        output['@truth accuracy']=self.binary_accuracy(
                self.predictions['all']['@truth'],
                self.targets['all']['@truth'])
        output['dollar amounts accuracy']=self.numerical_accuracy(
                self.predictions['all']['dollar_amounts'],
                self.targets['all']['dollar_amounts'])
        output['unigram jaccard']=self.unigram_jaccard(
                self.predictions['all']['other_out'],
                self.targets['all']['other_out'])
        output['string accuracy']=self.binary_accuracy(
                self.predictions['all']['other_in'],
                self.targets['all']['other_in'])
        output['info unified']=output['@truth accuracy']+output['dollar amounts accuracy']\
                +output['unigram jaccard']+output['string accuracy']
        if compute_confidence_intervals:
            output['@truth accuracy confidence interval']=\
                self.ninety_percent_confidence_interval(\
                output['@truth accuracy'])
            output['dollar amounts accuracy confidence interval']=\
                self.ninety_percent_confidence_interval(\
                output['dollar amounts accuracy'])
            output['unigram jaccard confidence interval']=\
                self.ninety_percent_confidence_interval(\
                output['unigram jaccard'])
            output['string accuracy confidence interval']=\
                self.ninety_percent_confidence_interval(\
                output['string accuracy'])
            output['info unified confidence interval']=\
                    self.ninety_percent_confidence_interval(\
                    output['info unified'])
        output['@truth accuracy']=self.mean(output['@truth accuracy'])
        output['dollar amounts accuracy']=self.mean(
                output['dollar amounts accuracy'])
        output['unigram jaccard']=self.mean(output['unigram jaccard'])
        output['string accuracy']=self.mean(output['string accuracy'])
        output['info unified']=self.mean(output['info unified'])
        return output

    def ninety_percent_confidence_interval(self,samples):
        # compute 90% confidence interval for average of samples
        # values for n degrees of freedom, two-sided.
        assert isinstance(samples,list)
        num_samples=len(samples)
        if num_samples<2:
            return None
        confidence=0.9
        t_value=scipy.stats.t.isf((1-confidence)/2,num_samples)
        mean=sum(samples)/num_samples
        S=sum([(x-mean)**2 for x in samples])/(num_samples-1)
        S=S**0.5
        confidence_interval=t_value*S/(num_samples**0.5)
        return confidence_interval

    def compute_metrics(self,compute_confidence_intervals=False):
        self.metrics=self.core(compute_confidence_intervals)
        info=self.info(compute_confidence_intervals)
        self.metrics.update(info)

    def print_metrics(self):
        def float2str(x):
            if isinstance(x,float):
                return str(round(x,4))
            return str(x)
        output=map(lambda x: x[0] + ': ' + float2str(x[1]), \
                sorted(self.metrics.items()))
        output='; '.join(output)
        return output

    def sorting_key(self):
        # return tuple of metrics by which to compare models
        # the idea for this ordering is that info is more
        # fine-grained than core,
        # also because it has access to more samples
        keys=['info unified','core unified']
        output=[]
        for key in keys:
            output.append(self.metrics[key])
        return tuple(output)


class Clustering:
    def __init__(self, data=None, params=None):
        for x in [data, params]:
            if x is not None:
                assert isinstance(x,list)
                assert all(isinstance(y,int) for y in x), str(x)
        self.data=data
        self.params=params

    def kmeans_cluster(self,num_clusters):
        # do k-means and return centroids and assignments
        kmeans=KMeans(n_clusters=num_clusters)
        X=np.array(self.data)
        X=np.expand_dims(X,axis=1)
        kmeans.fit(X)
        return kmeans

    def gm_cluster(self,num_clusters):
        gm=GaussianMixture(n_components=num_clusters, random_state=0)
        X=np.array(self.data)
        X=np.expand_dims(X,axis=1)
        gm.fit(X)
        return gm

    def predict_gmm(self,data):
        X=np.array(data)
        X=np.expand_dims(X,axis=1)
        posteriors=self.model.predict_proba(X)
        cluster_centers=[int(x[0]) for x in self.model.means_.tolist()]
        def prediction(weights):
            output=0
            for c,w in zip(cluster_centers,weights):
                output+=c*w
            return int(output)
        predictions=[prediction(x) for x in posteriors.tolist()]
        return posteriors,predictions

    def predict_kmeans(self,data):
        X=np.array(data)
        X=np.expand_dims(X,axis=1)
        assignments=self.model.predict(X)
        cluster_centers=[int(x[0]) for x in self.model.cluster_centers_.tolist()]
        predictions=[cluster_centers[x] for x in assignments.tolist()]
        return assignments,predictions

    def predict(self,data):
        if isinstance(self.model,KMeans):
            return self.predict_kmeans(data)
        elif isinstance(self.model,GaussianMixture):
            return self.predict_gmm(data)
        return None

    def get_centers(self):
        output=None
        if isinstance(self.model,KMeans):
            output=[int(x[0]) for x in self.model.cluster_centers_.tolist()]
        elif isinstance(self.model,GaussianMixture):
            output=[int(x[0]) for x in self.model.means_.tolist()]
        assert isinstance(output,list)
        assert all(isinstance(x,int) for x in output)
        return output

    def cluster(self):
        nclusters=1
        criterion=False
        while not criterion:
            nclusters*=2
            # kmeans cluster
            self.model=self.kmeans_cluster(nclusters)
            # check whether centroids cover the space appropriately
            accuracies=[]
            posteriors,predictions=self.predict(self.data)
            for val,pred in zip(self.data,predictions):
                error=self.numerical_accuracy(pred,val)
                accuracies.append(1 if error<1 else 0)
            criterion=all(x==1 for x in accuracies)
        # end while not criterion

    def numerical_accuracy(self,prediction,target):
        assert isinstance(prediction,int)
        assert isinstance(target,int)
        num=abs(prediction-target)
        denom=max(0.1*target,5000)
        return num/float(denom)

    def init_kmeans(self):
        kmeans=KMeans(n_clusters=len(self.params))
        kmeans.cluster_centers_=np.expand_dims(
                np.array([float(x) for x in self.params]),axis=1)

class DataLoader:
    def __init__(self,data,batch_size,epoch_size=None):
        self.data=data
        self.position=None
        assert isinstance(batch_size,int)
        assert batch_size > 0
        self.batch_size=batch_size
        self.epoch_size=epoch_size

    def find_date_surface_form(self,absolute_date_format,context):
        year,month,day=absolute_date_format.split('-')
        date=datetime.date(int(year),int(month),int(day))
        date_span=None
        for form in DATE_FORMATS:
            surface_form=date.strftime(form)
            if surface_form in context:
                date_span=surface_form
                break
        if date_span is None:
            return None
        elif date_span not in context:
            return None
        return date_span

    def make_data_point(self,case_ref,statute_ref,input_slots,outcome):
        data_point={}
        assert isinstance(statute_ref,str)
        assert ' ' not in statute_ref, statute_ref
        data_point['statute_ref']=statute_ref
        data_point['case_ref']=case_ref
        data_point['context']=self.data['cases'][case_ref]
        data_point['rules']=self.data['statutes'][statute_ref]
        data_point['spans']=self.data['spans'][statute_ref]
        input_slots=json.loads(input_slots) if isinstance(input_slots,str) else input_slots
        for k in input_slots:
            if isinstance(input_slots[k],list):
                input_slots[k]=input_slots[k][0]
            argument_type=get_argument_type(k,input_slots[k])
            if argument_type=='span' and (len(DATE_REGEXP.findall(str(input_slots[k])))>0):
                date_span=self.find_date_surface_form(input_slots[k],data_point['context'])
                if date_span is not None:
                    input_slots[k]=date_span
        data_point['input_slots']=input_slots
        outcome=json.loads(outcome) if isinstance(outcome,str) else outcome
        prompts=set()
        truth_included=False
        for k in outcome:
            truth_included=truth_included or k==TRUTH_KEY
            if isinstance(outcome[k],list):
                outcome[k]=outcome[k][0]
            argument_type=get_argument_type(k,outcome[k])
            if argument_type=='span' and (len(DATE_REGEXP.findall(str(outcome[k])))>0):
                date_span=self.find_date_surface_form(outcome[k],data_point['context'])
                if date_span is not None:
                    outcome[k]=date_span
            # only add spans if they appear in the context
            if (argument_type!='span') or (str(outcome[k]) in data_point['context']):
                prompts.add((k,outcome[k]))
        if not truth_included:
            prompts.add((TRUTH_KEY,))
        data_point['query_slots']=prompts
        return data_point


    def next_batch(self):
        batch={'context': [], 'rules': [], 'spans': [], 'case_ref': [], \
                'input_slots': [], 'query_slots': [], 'statute_ref': []}
        if self.position is None:
            self.position=0
        if self.position>=len(self.sample):
            return None
        batch_size=0
        while (len(batch['context'])<self.batch_size) \
                and (self.position<len(self.sample)):
            case_ref,statute_ref,input_slots,outcome=\
                    self.sample[self.position]
            if statute_ref not in self.data['statutes']:
                logging.debug('skipping ' + statute_ref)
                self.position+=1
                continue
            data_point=self.make_data_point(case_ref,statute_ref,\
                    input_slots,outcome)
            for key in ['statute_ref','case_ref', 'context','rules',\
                    'spans','input_slots','query_slots']:
                batch[key].append(data_point[key])
            batch_size+=1
            self.position+=1
        if batch_size==0: # check that returned batch wouldn't be empty
            return None
        return batch


    def get_integer_values(self):
        all_values=list()
        self.init_batches('train',get_all=True)
        batch=self.next_batch()
        while batch is not None:
            for queries in batch['query_slots']:
                for qq in queries:
                    arg_name,arg_value=qq
                    arg_type=get_argument_type(arg_name,arg_value)
                    if arg_type=='integer':
                        all_values.append(arg_value)
                    # end if arg_type=='integer'
                # end for qq in queries
            # end for queries in batch['query_slots']
            batch=self.next_batch()
        # end while batch is not None
        return all_values


class DataLoaderPretrain(DataLoader):
    def __init__(self,data,batch_size,epoch_size=25000):
        super().__init__(data,batch_size,epoch_size)
        self.data=data
        self.position=None
        assert isinstance(batch_size,int)
        assert batch_size > 0
        self.batch_size=batch_size
        # correct key mismatch
        def remap_keys(d):
            nu_d={}
            for k,v in d.items():
                if k=='tax':
                    key='tax'
                else:
                    key='s'+k.replace('(','_').replace(')','')
                nu_d[key]=v
            return nu_d
        self.data['statutes']=remap_keys(self.data['statutes'])
        self.data['spans']=remap_keys(self.data['spans'])
        # index cases
        data_size=len(self.data['train'])
        self.epoch_size=min(epoch_size,data_size)
        dev_size=int(0.1*self.epoch_size)
        cases={'pos': {}, 'neg': {}}
        for case in self.data['train']:
            stuff=case[-1]
            key='pos' if stuff['@TRUTH'] else 'neg'
            case_id=case[1]
            cases[key].setdefault(case_id,[])
            cases[key][case_id].append(case)
        # create dev set
        random.seed(2)
        dev=[]
        train={'pos': [], 'neg': []}
        counter={'pos': 0, 'neg': 0}
        pos_cases=sorted(cases['pos'].keys())
        neg_cases=sorted(cases['neg'].keys())
        random.shuffle(pos_cases); random.shuffle(neg_cases)
        while len(dev)<dev_size:
            if counter['neg']<counter['pos']:
                key='neg'
                if len(neg_cases)==0:
                    break
                case=neg_cases.pop()
            else:
                key='pos'
                if len(pos_cases)==0:
                    break
                case=pos_cases.pop()
            dev.extend(cases[key][case])
            counter[key]+=len(cases[key][case])
        for key,_cases in zip(['pos','neg'],[pos_cases,neg_cases]):
            for case in _cases:
                train[key].extend(cases[key][case])
        assert len(train['pos'])+len(train['neg'])+len(dev)==data_size, \
                str((len(train['pos'])+len(train['neg']), \
                len(dev), data_size))
        self.data['train']=train
        self.data['dev']=dev
        self.data['test']=data['test']['gold']
        logging.info('data size: ' + 'train pos: ' + str(len(self.data['train']['pos'])) \
                + '; train neg ' + str(len(self.data['train']['neg'])) \
                + '; dev ' + str(len(self.data['dev'])) \
                + '; test ' + str(len(self.data['test'])))

    def init_batches(self,split,get_all=False):
        self.position=None
        if split=='train':
            random.shuffle(self.data['train']['pos'])
            random.shuffle(self.data['train']['neg'])
            size=int(self.epoch_size/2)
            def sample_with_replacement(pop,num):
                output=[]
                while len(output)<num:
                    output.append(random.choice(pop))
                return output
            if self.epoch_size<(len(self.data['train']['pos'])+len(self.data['train']['neg'])):
                self.sample=sample_with_replacement(\
                        self.data['train']['pos'],size)
                self.sample=self.sample+sample_with_replacement(\
                        self.data['train']['neg'],size)
            else:
                self.sample=self.data['train']['pos']+self.data['train']['neg']
            if get_all:
                self.sample=self.data['train']['pos']+self.data['train']['neg']
            random.shuffle(self.sample)
        elif split=='dev':
            self.sample=self.data['dev']
        elif split=='test':
            self.sample=self.data['test']
        else:
            self.sample=None

class DataLoaderTrain(DataLoader):
    def __init__(self,data,batch_size):
        super().__init__(data,batch_size)
        self.data=data
        self.position=None
        assert isinstance(batch_size,int)
        assert batch_size > 0
        self.batch_size=batch_size
        # correct key mismatch
        def remap_keys(d):
            nu_d={}
            for k,v in d.items():
                if k=='tax':
                    key='tax'
                else:
                    key='s'+k.replace('(','_').replace(')','')
                nu_d[key]=v
            return nu_d
        self.data['statutes']=remap_keys(self.data['statutes'])
        self.data['spans']=remap_keys(self.data['spans'])
        # index automatic cases
        cases={'pos': [], 'neg': []}
        # remove any cases that appear in the dev set from the training set
        dev_cases=[ x[1] for x in self.data['dev'] ]
        for case in self.data['train']['automatic']:
            if case[1] in dev_cases:
                continue
            stuff=case[-1]
            key='pos' if stuff['@TRUTH'] else 'neg'
            cases[key].append(case)
        self.automatic_data=cases

    def init_batches(self,split,get_all=True):
        self.position=None
        if split=='train':
            self.sample=self.data['train']['gold']
            random.shuffle(self.sample)
        elif split=='dev':
            self.sample=self.data['dev']
        elif split=='test':
            self.sample=self.data['test']
        else:
            self.sample=None


class MixedModel(nn.Module):
    def __init__(self,pretrained_model='bert-base-uncased',
            tokenizer='bert-base-uncased',max_length=512,
            data_loader=None,max_depth=5,gmm_model=None):
        super(MixedModel, self).__init__()
        self.pretrained_model=pretrained_model
        # BERT base
        try:
            self.bert=AutoModel.from_pretrained(pretrained_model)
        except:
            self.bert=BertModel.from_pretrained(pretrained_model)
        self.cluster_model=gmm_model
        # lighten the load of the computation
        self.bert.config.output_hidden_states=False
        self.bert.config.output_attentions=False
        # LSTM with attention over context to predict value
        input_size=self.bert.config.hidden_size # previous output (using BERT embeddings)
        vocab_size=self.bert.config.vocab_size
        numerical_vocab=self.cluster_model.get_centers()
        self.numerical_weights=torch.cuda.FloatTensor(numerical_vocab)
        self.numerical_classifier=nn.Linear(8*self.bert.config.hidden_size,
                                           len(numerical_vocab))
        self.left_boundary=nn.Linear(4*self.bert.config.hidden_size,1)
        self.right_boundary=nn.Linear(4*self.bert.config.hidden_size,1)
        self.truth_classifier=nn.Linear(self.bert.config.hidden_size,1)
        if isinstance(tokenizer,str):
            self.tokenizer=BertTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer=tokenizer
        self.bert_max_length=max_length
        self.gpu=False
        self.data_loader=data_loader
        self.max_depth=max_depth
        '''
        training stage  teacher forcing for decode  use of structure
        0               yes                         no
        1               no                          no
        2               no                          yes
        '''
        self.training_stage=0
        self.unit_tests()

    def cuda(self):
        self.gpu=True
        self.bert.to('cuda')
        self.numerical_classifier.to('cuda')
        self.left_boundary.to('cuda')
        self.right_boundary.to('cuda')
        self.truth_classifier.to('cuda')

    def freeze_bert(self, thaw_top_layer=False):
        for param in self.bert.parameters():
            param.requires_grad=False
        if thaw_top_layer:
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad=True

    def set_training_stage(self,training_stage):
        self.training_stage=training_stage
        if self.training_stage==2:
            self.freeze_bert(thaw_top_layer=False)

    def unit_tests(self):
        # run a bunch of unit tests for methods
        assert self.test_grounding()

    def test_grounding(self):
        return True

    def to_dollar_amount(self,n):
        output=str(n)[::-1]
        output=','.join(output[ii:ii+3] for ii in range(0,len(output),3))
        output='$'+output[::-1]
        return output

    def ground_rule(self,groundings,spans,rule):
        ''' groundings is dict str -> str mapping name of span
        to its string value
        spans is a dict of str -> list where list contains
        pairs of (start, stop) character indices
        of where the given span is to be found in rule
        rule is a string '''
        for g in groundings:
            gr=groundings[g]
            assert not isinstance(gr,list), str(gr)
        for g in groundings:
            gr=str(groundings[g])
            if g not in spans:
                continue
            spans_to_ground=sorted(spans[g])
            for ii,(start,stop) in enumerate(spans_to_ground):
                # ground
                rule=rule[:start]+str(gr)+rule[stop+1:]
                # adjust other spans in same grounding
                for jj in range(ii+1,len(spans_to_ground)):
                    start1,stop1=spans_to_ground[jj]
                    if start1>stop:
                        nu_start1=start1-stop+len(gr)+start-1
                        nu_stop1=len(gr)+start-stop+stop1-1
                        spans_to_ground[jj]=(nu_start1, nu_stop1)
                # adjust other spans
                for g1 in spans:
                    if g==g1:
                        continue
                    other_spans=set(spans[g1])
                    for start1,stop1 in other_spans:
                        if (start1>=start) and (stop1<=stop):
                            spans[g1].remove((start1,stop1))
                        if start1>stop:
                            nu_start1=start1-stop+len(gr)+start-1
                            nu_stop1=len(gr)+start-stop+stop1-1
                            spans[g1].remove((start1,stop1))
                            spans[g1].add((nu_start1, nu_stop1))
            # end for ii,(start,stop) in enumerate(spans_to_ground)
            del spans[g]
            # some spans might have disappeared
            to_remove=set(filter(lambda x: len(spans[x])==0, spans.keys()))
            for x in to_remove:
                del spans[x]
        # end for g in groundings
        return rule,spans

    def tokenize(self,rule,spans):
        ''' rule is a string to be tokenized
        spans is a dict of str -> set which contains
        pairs of character-level spans
        output is tokens (list of tokens)
        and span_masks, a dict of str -> list where each list
        is a binary mask of where the span appears in the token
        list '''
        # tokenize plain rule
        tokens=self.tokenizer.tokenize(rule)
        # create alignment map source word index -> target token index
        src_ind,tgt_ind=0,0
        buff=""
        src=rule
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
            logging.debug('problem with ' + str(tgt_side) \
                    + ' vs ' + str(len(tgt)))
        # make target mask
        span_masks={}
        for name in spans:
            span_masks[name]=[ 0 for _ in tokens ]
            for start,stop in spans[name]:
                problem=(start<0) or (stop<0) or (start>=len(mapping))\
                        or (stop>=len(mapping)) or (start>stop)
                if problem:
                    logging.debug('problem with ' + str(spans[name]))
                    logging.debug('mapping: ' + str(mapping))
                    logging.debug('len(mapping): ' + str(len(mapping)))
                    logging.debug('rule: ' + str(rule))
                    logging.debug('tokens: ' + str(tokens))
                    logging.debug('spans: ' + str(spans))
                    start,stop=0,len(mapping)-1 # default values so it keeps going
                    while mapping[start] is None:
                        start+=1
                    while mapping[stop] is None:
                        stop-=1
                start_tok=mapping[start]
                if start_tok is None: # we hit a space
                    start_tok=mapping[start+1]
                stop_tok=mapping[stop]
                if stop_tok is None: # we hit a space
                    stop_tok=mapping[stop-1]
                if (start_tok is None) and (stop_tok is None):
                    all_ints=list(filter(lambda x: x is not None, mapping))
                    if len(all_ints)==0:
                        mapping=[0]
                    start_tok = min(all_ints)
                    stop_tok = max(all_ints)
                elif stop_tok is None:
                    stop_tok = start_tok
                elif start_tok is None:
                    start_tok = stop_tok
                for ii in range(start_tok,stop_tok+1):
                    span_masks[name][ii]=1
            assert max(span_masks[name])==1, str(name)
        return tokens, span_masks

    def compute_queries(self,queries,span_masks):
        ''' queries is a list of set, each containing tuples
        of either (span name, value) or (span name, None)
        span_masks is a list of dict, each dict containing the
        mapping from span name to span mask
        output is a list of (index, span name, query_mask),
        where index refers to an item in the batch and
        query_mask is a binary mask of the span whose value
        is being queried
        if a value is specified, return
        (index, span name, query_mask, value) '''
        output=[]
        for ii,q in enumerate(queries):
            if len(q)>0:
                item=sorted(q)[-1]
                assert len(item) in [1,2], str(item)
                name=item[0]
                if name==TRUTH_KEY:
                    continue # that one is done at last and separately
                value=None
                if len(item)==2:
                    value=item[1]
                if name not in span_masks[ii]:
                    continue # the query may have been grounded previously
                mask=span_masks[ii][name]
                if value is None:
                    output.append((ii,name,mask))
                else:
                    output.append((ii,name,mask,value))
        return output

    def compute_bert_representations_no_query(self,batch):
        # compute bert representations for the whole batch
        # and don't compute query masks
        text_input=[(_c,_r) for _c,_r in \
                zip(batch['context'],batch['rules'])]
        input_dicts = [self.tokenizer.encode_plus(x, \
                text_pair=y, add_special_tokens=True, \
                truncation_strategy='only_second', \
                max_length=self.bert_max_length) for x,y in text_input]
        input_ids=[x['input_ids'] for x in input_dicts]
        token_type_ids=[x['token_type_ids'] for x in input_dicts]
        input_mask = [[1]*len(x) for x in input_ids]
        input_ids = list(map(lambda x: \
                x+[0]*(self.bert_max_length-len(x)) \
                if len(x)<self.bert_max_length else x, input_ids))
        token_type_ids = list(map(lambda x: \
                x+[0]*(self.bert_max_length-len(x)) \
                if len(x)<self.bert_max_length else x, token_type_ids))
        input_mask = list(map(lambda x: \
                x+[0]*(self.bert_max_length-len(x)) \
                if len(x)<self.bert_max_length else x, input_mask))
        # turn into tensor + move to gpu
        input_tensor=torch.LongTensor(input_ids).to('cuda')
        token_type_ids_tensor=torch.LongTensor(token_type_ids).to('cuda')
        mask_tensor=torch.FloatTensor(input_mask).to('cuda')
        assert torch.equal(torch.gt(input_tensor,0).byte(),\
                mask_tensor.byte())
        # run BERT
        assert input_tensor.size(0)>0, str(input_tensor) + ' and ' + str(text_input)
        assert input_tensor.size(1)>0, str(input_tensor) + ' and ' + str(text_input)
        assert mask_tensor.size(1)>0, str(mask_tensor) + ' and ' + str(text_input)
        assert mask_tensor.size(1)>0, str(mask_tensor) + ' and ' + str(text_input)
        representations,_=self.bert(input_tensor,\
                token_type_ids=token_type_ids_tensor, \
                attention_mask=mask_tensor) # batch x len x dim
        return representations

    def compute_bert_representations(self,batch,queries=None):
        if queries is None:
            return self.compute_bert_representations_no_query(batch)
        text_input=[(batch['context'][item[0]],\
                batch['rules'][item[0]]) for item in queries]
        span_masks=[q[2] for q in queries]
        input_dicts = [self.tokenizer.encode_plus(x, \
                text_pair=y, add_special_tokens=True, \
                truncation_strategy='only_second', \
                max_length=self.bert_max_length) for x,y in text_input]
        input_ids = [x['input_ids'] for x in input_dicts]
        token_type_ids = [x['token_type_ids'] for x in input_dicts]
        input_mask = [[1]*len(x) for x in input_ids]
        context_mask = [ [1-x for x in y] for y in token_type_ids ]
        # pad
        for ii,mask in enumerate(span_masks):
            # pad on the left
            span_masks[ii] = \
                [0]*(len(input_ids[ii])-len(mask)) + mask \
                if len(mask)<len(input_ids[ii]) else mask
            span_masks[ii] = span_masks[ii][:self.bert_max_length]
            assert len(span_masks[ii]) == len(token_type_ids[ii]), \
                    str(len(span_masks[ii])) + ' vs ' \
                    + str(len(token_type_ids[ii]))
            span_masks[ii] = list( min(x,y) for x,y in \
                    zip(span_masks[ii], token_type_ids[ii]) )
            # pad on the right
            _x=span_masks[ii]
            span_masks[ii]=_x+[0]*(self.bert_max_length-len(_x)) \
                    if len(_x)<self.bert_max_length else _x
        # end for ii,mask in enumerate(span_masks)
        input_ids = list(map(lambda x: \
                x+[0]*(self.bert_max_length-len(x)) \
                if len(x)<self.bert_max_length else x, input_ids))
        token_type_ids = list(map(lambda x: \
                x+[0]*(self.bert_max_length-len(x)) \
                if len(x)<self.bert_max_length else x, token_type_ids))
        input_mask = list(map(lambda x: \
                x+[0]*(self.bert_max_length-len(x)) \
                if len(x)<self.bert_max_length else x, input_mask))
        context_mask = list(map(lambda x: \
                x+[0]*(self.bert_max_length-len(x)) \
                if len(x)<self.bert_max_length else x, context_mask))
        # turn into tensor + move to gpu
        input_tensor=torch.LongTensor(input_ids).to('cuda')
        token_type_ids_tensor=torch.LongTensor(token_type_ids).to('cuda')
        mask_tensor=torch.FloatTensor(input_mask).to('cuda')
        context_mask_tensor=torch.ByteTensor(context_mask).to('cuda')
        assert torch.equal(torch.gt(input_tensor,0).byte(),\
                mask_tensor.byte())
        for ii,mask in enumerate(span_masks):
            span_masks[ii] = torch.LongTensor(span_masks[ii]).to('cuda')
        span_masks=torch.stack(span_masks,dim=0)
        # run BERT
        representations,_=self.bert(input_tensor,\
                token_type_ids=token_type_ids_tensor, \
                attention_mask=mask_tensor) # batch x len x dim
        return representations, span_masks, context_mask_tensor

    def compute_joined_representations(self, \
            bert_representations, span_masks):
        span_masks=torch.unsqueeze(span_masks,dim=2).byte()
        # generate embeddings of current queries
        weights=torch.unsqueeze(bert_representations,dim=1)\
                *torch.unsqueeze(bert_representations,dim=2) # dot products
        weights=torch.sum(weights,dim=3)
        # sometimes the context is too long for the rule to appear in it, meaning the
        # corresponding span mask is filled with 0s. This will make nans appear after
        # the softmax.
        weights=weights.masked_fill((1-span_masks).bool(),-float('inf'))
        weights=torch.softmax(weights,dim=1) # normalize
        # 'failed' softmaxes come from everything being equal to -inf,
        # so can be replaced with 0 after the softmax
        weights=weights.masked_fill(torch.isnan(weights),0)
        assert not torch.any(torch.isnan(weights))
        placeholder_reps=torch.unsqueeze(bert_representations,dim=2)\
                *torch.unsqueeze(weights,dim=3)
        placeholder_reps=torch.sum(placeholder_reps,\
                dim=1) # this contains the s_j
        joined_reps=torch.cat([placeholder_reps,bert_representations,\
                torch.abs(placeholder_reps-bert_representations),\
                placeholder_reps*bert_representations],dim=2)
        return joined_reps

    def detokenize_bert(self,tokens):
        # turn 'tokens', a list of tokens, back into a string of words
        # this process is approximate
        output=' '.join(tokens)
        output=output.replace(' ##', '')
        for p in [',',')','.',"'"]:
            output=output.replace(' '+p,p)
        for p in ['(','$',"'"]:
            output=output.replace(p+' ',p)
        if output=='[CLS]': # this means that the answer could not be found in the context
            output=''
        return output

    def span_finder(self,features,context_mask,context_tokens):
        """ context are the joined reps, so already conditioned
        on the target argument
        this runs a squad-like model """
        # only run on the context part of the case, not the statute
        seq_mask=(1-torch.unsqueeze(context_mask,dim=2)).bool()
        # features is (batch, length, dim)
        left_logits=self.left_boundary(features) # (batch, length, 1)
        left_logits=left_logits.masked_fill(seq_mask,-float('inf'))
        right_logits=self.right_boundary(features) # (batch, length, 1)
        right_logits=right_logits.masked_fill(seq_mask,-float('inf'))
        # predictions
        # this contains the prob of a span from i to j at position (i,j)
        joint_spans=left_logits+torch.transpose(right_logits,1,2) # (batch, length, length)
        # find best span such that j>=i
        diagonal_mask=(1-torch.triu(torch.ones_like(joint_spans))).bool()
        joint_spans.masked_fill_(diagonal_mask,-float('inf'))
        start=torch.argmax(torch.max(joint_spans,dim=2,keepdim=True)[0],dim=1)
        stop=torch.argmax(torch.max(joint_spans,dim=1,keepdim=True)[0],dim=2)
        if not (start<=stop).all():
            logging.debug('problem with start and stop')
        start=start.tolist()
        stop=stop.tolist()
        assert len(start)==len(stop)
        assert len(start)==len(context_tokens)
        predictions=[]
        context_tokens_augmented=[['[CLS]']+x+['[SEP]'] for x in context_tokens]
        assert all(torch.tensor([len(x) for x in context_tokens_augmented])==torch.sum(context_mask,dim=1))
        for aa,bb,cc in zip(start,stop,context_tokens_augmented):
            start_=aa[0]
            stop_=max(bb[0],start_)
            pred=self.detokenize_bert(cc[start_:stop_+1])
            predictions.append(pred)
        left_logits=torch.squeeze(left_logits,dim=2) # (batch, length)
        right_logits=torch.squeeze(right_logits,dim=2) # (batch, length)
        return (left_logits, right_logits), predictions

    def numerical_predictor(self,features,context_mask,context_tokens):
        """ context are the joined reps, so already conditioned
        on the target argument
        this runs a numerical predictor, which is a multiclass classifier """
        avg_pool=torch.mean(features,dim=1) # (batch, dim)
        max_pool,_=torch.max(features,dim=1) # (batch, dim)
        seq_features=torch.cat([avg_pool,max_pool],dim=1) # (batch, 2*dim)
        logits=self.numerical_classifier(seq_features) # (batch, num categories)
        logits=torch.sum(torch.nn.functional.softmax(logits,dim=1) \
                *torch.unsqueeze(self.numerical_weights,dim=0),dim=1) # (batch, 1)
        predictions=logits.int()
        return logits, predictions

    def binary_predictor(self,features,context_mask,context_tokens):
        """ context are the BERT reps
        this runs an NLI-like model """
        embeddings=features[:,0,:]
        truth_logits=self.truth_classifier(embeddings)
        truth_predictions=(truth_logits>0).cpu().tolist()
        return truth_logits, truth_predictions

    def infer(self,queries,features,context_mask,context_tokens,grad=False):
        # context_mask is meant to indicate where the content is in context
        all_logits={}
        all_predictions={}
        tokens=[context_tokens[q[0]] for q in queries]
        assert len(tokens)==features.size(0)
        # make predictions with each model
        for model_name,model in [('span_finder',self.span_finder), \
                ('numerical_predictor',self.numerical_predictor)]:
            l,p = model(features,context_mask,tokens)
            all_logits[model_name]=l
            all_predictions[model_name]=p
        # pick and choose based on argument type
        logits,predictions=[],[]
        for ii,q in enumerate(queries):
            if grad:
                assert len(q)==4
                argument_type=get_argument_type(q[1],q[3])
            else:
                argument_type=get_argument_type(q[1])
            assert argument_type in ['integer', 'span']
            if argument_type=='span':
                logits.append((all_logits['span_finder'][0][ii,:],\
                        all_logits['span_finder'][1][ii,:]))
                predictions.append(all_predictions['span_finder'][ii])
            elif argument_type=='integer':
                logits.append(all_logits['numerical_predictor'][ii])
                predictions.append(all_predictions['numerical_predictor'][ii])
        # the logits of various models are not compatible so this must be kept in a list
        assert len(logits)==len(queries)
        assert len(predictions)==len(queries)
        return logits, predictions

    def forward_single(self,batch,grad=False):
        batch.setdefault('logits',[ {} for _ in batch['query_slots'] ])
        batch.setdefault('predictions',[ {} for _ in batch['query_slots'] ])
        for key in ['logits','predictions']:
            for ii,x in enumerate(batch[key]):
                batch[key][ii] = batch[key][ii] if batch[key][ii] is not None else {}
                assert batch[key][ii] is not None, str(key)
        batch['queries']=[None for _ in batch['query_slots']] # make a copy
        for ii,q in enumerate(batch['query_slots']):
            batch['queries'][ii]=copy.deepcopy(q)
        # initial grounding and span mask generation
        batch['groundings']=copy.deepcopy(batch['input_slots'])
        for ii,s in enumerate(batch['spans']):
            span_dict={}
            for span in batch['spans'][ii]:
                start,stop,name=span
                span_dict.setdefault(name,set())
                span_dict[name].add((start,stop))
            batch['spans'][ii]=span_dict
        # end for ii,_ in enumerate(batch['spans'])
        while True:
            # 1. ground out rules and create masks
            batch['tokens'] = [None for _ in batch['rules']]
            batch['span_masks'] = [None for _ in batch['rules']]
            for ii,_ in enumerate(batch['groundings']):
                batch['rules'][ii],batch['spans'][ii]=\
                        self.ground_rule(batch['groundings'][ii],\
                        batch['spans'][ii],batch['rules'][ii])
                batch['tokens'][ii],batch['span_masks'][ii]=\
                        self.tokenize(batch['rules'][ii],\
                        batch['spans'][ii])
            # end for ii,_ in enumerate(batch['groundings'])
            # 2. run model over relevant queries
            # list queries and compute masks
            queries=self.compute_queries(batch['queries'],\
                    batch['span_masks'])
            if len(queries)==0:
                break # means we can move to last step of computing truth
            # compute bert reps
            bert_representations,span_masks,context_masks=\
                    self.compute_bert_representations(batch,queries)
            # compute placeholder reps and joined reps
            joined_reps = self.compute_joined_representations(\
                    bert_representations, span_masks)
            # compute gold output
            # decode with attention
            context_tokens=[self.tokenizer.tokenize(x) for x in batch['context']]
            logits,predictions=\
                    self.infer(queries,joined_reps,context_masks,context_tokens,grad)
            # insert logits where appropriate
            assert len(queries)==len(logits)
            for index,q in enumerate(queries):
                name=q[1]
                batch_index=q[0]
                batch['logits'][batch_index][name]=logits[index]
                batch['predictions'][batch_index][name]=predictions[index]
            # delete queries that were answered
            # and insert prediction into input slots
            for ii,_ in enumerate(batch['queries']):
                _queries=copy.deepcopy(batch['queries'][ii])
                for name in batch['predictions'][ii]:
                    for item in _queries:
                        if name!=item[0]:
                            continue
                        batch['queries'][ii].remove(item)
                        # when in training mode and in first
                        # training stage, use gold value
                        if grad and (self.training_stage==0):
                            grounding=item[1]
                        else:
                            grounding=batch['predictions'][ii][name]
                        assert not isinstance(grounding,list)
                        batch['groundings'][ii][item[0]]=grounding
                    # end for item in _queries
                # end for name in batch['predictions'][ii]
            # end for ii,_ in enumerate(batch['queries'])
        # end while True
        # final pass to compute truth value of whole thing
        # 1. ground out rules and create masks
        batch['tokens'] = [None for _ in batch['rules']]
        batch['span_masks'] = [None for _ in batch['rules']]
        for ii,_ in enumerate(batch['groundings']):
            batch['rules'][ii],batch['spans'][ii]=\
                    self.ground_rule(batch['groundings'][ii],\
                    batch['spans'][ii],batch['rules'][ii])
            batch['tokens'][ii],batch['span_masks'][ii]=\
                    self.tokenize(batch['rules'][ii],\
                    batch['spans'][ii])
        # end for ii,_ in enumerate(batch['groundings'])
        # 2. compute bert reps
        bert_representations=\
                self.compute_bert_representations(batch)
        # 3. predict truth value using [CLS]
        truth_logits,truth_predictions=self.binary_predictor(bert_representations,None,None)
        for ii,_ in enumerate(batch['logits']):
            batch['logits'][ii][TRUTH_KEY]=truth_logits[ii,:]
            batch['predictions'][ii][TRUTH_KEY]=truth_predictions[ii][0]

    def get_children(self,data_point,structure):
        # structure is a list of strings, with exactly one rule per line
        assert isinstance(data_point,dict)
        output=None
        for item in structure:
            head=item.split(':-')[0].strip()
            predicate_name=head.split('(')[0]
            if predicate_name==data_point['statute_ref']:
                output=item
        assert output is not None, \
                "error with " + str(data_point['statute_ref'])
        # remove spaces around commas and colons
        while '  ' in output:
            output=output.replace('  ',' ')
        output=output.replace(', ',',')
        output=output.replace(' ,',',')
        output=output.replace(' :',':')
        output=output.replace(': ',':')
        return output

    def inverse_polish_notation_rec(self,body):
        assert isinstance(body,str), str(body)
        level=0
        ops={}
        body_in=body.strip(' ')
        logging.debug('body in: ' + str(body_in))
        if (body_in[0]=='[') and (body_in[-1]==']'):
            body_in=body_in[1:-1].strip(' ')
        for w in body_in.split(' '):
            if w in ['AND','OR']:
                ops[level]=w
            if w=='[':
                level+=1
            if w==']':
                level-=1
        main_op=ops.get(0)
        logging.debug('main op: ' + str(main_op))
        if main_op is None:
            if body_in.startswith('NOT'):
                body_in=body_in.strip(' ')[4:].strip(' ')
                output = self.inverse_polish_notation_rec(body_in)\
                        + [('NOT',1)]
            else:
                output = [body_in.strip(' ')]
        else:
            output=[]
            children=[]
            buff=[]
            level=0
            for w in body_in.split(' '):
                if (w==main_op) and (level==0):
                    children.append(' '.join(buff))
                    buff=[]
                else:
                    buff.append(w)
                    if w=='[':
                        level+=1
                    if w==']':
                        level-=1
            if len(buff)>0:
                children.append(' '.join(buff))
            logging.debug('children: ' + str(children))
            for x in children:
                output.extend(self.inverse_polish_notation_rec(x))
            output.append( (main_op,len(children)) )
        return output

    def inverse_polish_notation(self,dependencies):
        head=dependencies.split(':-')[0].strip(' ')
        body=dependencies.split(':-')[1].strip(' ')
        stack=self.inverse_polish_notation_rec('[ '+body+' ]')
        logging.debug('dependencies: ' + str(dependencies))
        logging.debug('stack: ' + str(stack))
        return stack

    def add_data(self,data_point,stack):
        # propagate context, groundings etc to the stack
        case_ref=data_point['case_ref']
        base_input_slots=data_point['input_slots']
        parent_statute_ref=data_point['statute_ref']
        for ii,x in enumerate(stack):
            if isinstance(x,tuple): # means it's an operator
                continue
            statute_ref=x.split('(')[0]
            args=x.split('(')[1].split(')')[0].split(',')
            input_slots={}
            query_slots=set()
            for a in args:
                if ':' in a:
                    original,foreign=a.split(':')
                    if foreign in base_input_slots:
                        input_slots[original]=copy.deepcopy(base_input_slots[foreign])
                    else:
                        query_slots.add((original,))
                else:
                    if a in base_input_slots:
                        input_slots[a]=copy.deepcopy(base_input_slots[a])
                    else:
                        query_slots.add((a,))
            y=self.data_loader.make_data_point(case_ref,\
                    statute_ref,input_slots,'{}')
            y['query_slots']=query_slots
            y['parent']=parent_statute_ref
            stack[ii]=y

    def do_operation(self, operation, inputs, dependencies=None):
        # check that they share the same parent
        parents=[x.get('parent') for x in inputs]
        assert len(set(parents))==1, str(parents)
        if operation=='NOT':
            return self.negate(inputs)
        if operation=='AND':
            return self.conjunction(inputs, dependencies)
        if operation=='OR':
            return self.disjunction(inputs, dependencies)

    def negate(self,inputs):
        assert len(inputs)==1
        output=inputs[0]
        # negate truth value
        output['logits'][TRUTH_KEY] = 1-output['logits'][TRUTH_KEY]
        output['predictions'][TRUTH_KEY] = 1-output['predictions'][TRUTH_KEY]
        # drop all other groundings etc since you assume you failed to
        # prove it
        for key1 in ['logits','predictions','groundings']:
            keys2=list(output[key1].keys())
            if TRUTH_KEY in keys2:
                keys2.remove(TRUTH_KEY)
            for key2 in keys2:
                del output[key1][key2]
        return output

    def create_mapping(self,dependencies):
        # mapping has same order as inputs in aggregate_in_order
        # that's important because the same predicate might appear
        # twice but with a different mapping
        mapping=[]
        for w in dependencies:
            if not isinstance(w,str):
                continue
            mapping.append({})
            args=w.split('(')[1].rstrip(')').split(',')
            for a in args:
                if ':' in a:
                    origin,destination=a.split(':')
                else:
                    origin,destination=a,a
                mapping[-1][origin]=destination
        return mapping

    def aggregate_in_order(self,order,inputs,dependencies):
        logging.debug('aggregation input'.upper())
        for x in inputs:
            logging.debug(x['logits'].keys())
        output={}
        mapping=self.create_mapping(dependencies)
        for ii in order:
            if ('parent' not in output) and ('parent' in inputs[ii]):
                output['parent']=inputs[ii]['parent']
            for key in ['logits','predictions','groundings']:
                output.setdefault(key,{})
                for slot_key in inputs[ii][key]:
                    if slot_key==TRUTH_KEY:
                        destination_key=TRUTH_KEY
                    else:
                        destination_key=mapping[ii].get(slot_key)
                    if destination_key is None:
                        continue # means the key is irrelevant to parent statute; happens eg in s151a
                    if destination_key not in output[key]:
                        output[key][destination_key]=inputs[ii][key][slot_key]
                # end for slot_key in inputs[ii][key]
            # end for key in ['logits','predictions','groundings']
        # end for ii in order
        logging.debug('aggregation output'.upper())
        logging.debug(output['logits'].keys())
        return output

    def conjunction(self,inputs,dependencies):
        logits=[float(y['logits'][TRUTH_KEY].cpu()) for y in inputs]
        order=sorted(range(len(inputs)), key=lambda x: logits[x])
        return self.aggregate_in_order(order,inputs,dependencies)

    def disjunction(self,inputs,dependencies):
        logits=[float(y['logits'][TRUTH_KEY].cpu()) for y in inputs]
        order=sorted(range(len(inputs)),
                key=lambda x: logits[x],reverse=True)
        order=[order[0]] # kind of an exclusive or
        return self.aggregate_in_order(order,inputs,dependencies)

    def update_slots(self, input_data_point, data_point):
        # backward mapping was done in previous aggregation step
        logging.debug(input_data_point['logits'].keys())
        logging.debug(data_point['spans'])
        data_point_slots=set(x[2] for x in data_point['spans'])
        for key in ['groundings','predictions']:
            for g,v in input_data_point[key].items():
                if (g in data_point_slots) and \
                        (g not in data_point['input_slots']):
                    data_point['input_slots'][g]=copy.deepcopy(v)
        # save logits for backprop
        data_point['logits']={}
        data_point['predictions']={}
        for g in input_data_point['logits']:
            data_point['logits'][g]=input_data_point['logits'][g]
            data_point['predictions'][g]=input_data_point['predictions'][g]
        assert data_point['logits'].keys()==data_point['predictions'].keys()
        logging.debug(data_point['logits'].keys())

    def to_batch(self,batches):
        # convert batches of size 1 to a single batch
        output={}
        for ii,d in enumerate(batches):
            for k,v in d.items(): # convert to batch of size 1
                output.setdefault(k,[None for _ in batches])
                output[k][ii]=v
        for k,v in output.items():
            assert len(v)==len(batches)
        return output

    def from_batch(self,batch):
        # convert batch to batches of size 1
        output=[]
        for k,v in batch.items():
            for ii,_v in enumerate(v):
                if len(output)<=ii:
                    output.append({})
                assert k not in output[ii]
                output[ii][k]=_v
        for k,v in batch.items():
            assert len(output)==len(v)
        return output

    def check_stack(self,stack):
        is_ok=True
        for x in stack:
            is_ok=is_ok and (isinstance(x,tuple) or isinstance(x,dict))
        return is_ok

    def forward_stack(self,starting_point):
        stacks=[ [x] for x in self.from_batch(starting_point)]
        # does not quite reflect depth of structure but number of
        # times we've expanded the statute into its dependencies.
        # One expansion can lead to increase in structure depth by
        # more than 1
        depth_tracker=[ [0] for _ in stacks]
        logging.debug('starting point: ' + str(starting_point))
        positions=[ 0 for _ in stacks]
        done_tracker= [ False for _ in stacks ]
        while not all(done_tracker):
            states=[ None for _ in stacks ] # track which kind of operation is going on
            # operations before forwarding through model
            batched_operations=[]
            for stacki,stack in enumerate(stacks):
                if done_tracker[stacki]:
                    continue
                position=positions[stacki]
                stack_sketch=[(x.get('statute_ref','unk'),x.get('parent')) \
                        if isinstance(x,dict) else x \
                        for x in stacks[stacki]]
                logging.debug('stack: ' +str(stacki) + '; position: ' + str(position)\
                        + '; stack: ' \
                        + str(stack_sketch))
                assert self.check_stack(stack)
                if position>=len(stack):
                    assert len(stack)==1
                    done_tracker[stacki]=True
                    continue # means that last part is resolved, and stack should only contain a single item
                if isinstance(stacks[stacki][position],tuple):
                    operation,arity=stacks[stacki][position]
                    if operation=='LOAD':
                        states[stacki]=0 
                    elif operation=='NOT':
                        states[stacki]=1 
                    else:
                        states[stacki]=2
                else:
                    dependencies=self.get_children(stacks[stacki][position],\
                            self.data_loader.data['structure'])
                    if (len(dependencies.split(':-'))==1) \
                            or (depth_tracker[stacki][position]==self.max_depth):
                        states[stacki]=3
                    else: # take into account dependencies
                        states[stacki]=4
                assert states[stacki] in range(5)
                if states[stacki]==0:
                    self.update_slots(stacks[stacki][position-1],\
                            stacks[stacki][position+1])
                    logging.debug('in forward_stack'.upper())
                    logging.debug(stacks[stacki][position+1]['logits'].keys())
                    # then make final pass forward to resolve remaining
                    # text in statute
                    batched_operations.append((stacki,position+1))
                elif states[stacki]==1:
                    operation,arity=stacks[stacki][position]
                    stacks[stacki][position]=self.do_operation(operation,\
                            stacks[stacki][position-arity:position])
                    del stacks[stacki][position-arity:position]
                    del depth_tracker[stacki][position-arity:position]
                    positions[stacki]+=1-arity
                elif states[stacki]==2:
                    operation,arity=stacks[stacki][position]
                    logging.debug('operation, arity: ' + str((operation,arity)))
                    logging.debug('look for parent START')
                    parent=None
                    for thing in stacks[stacki][position-arity:position]:
                        logging.debug(thing.get('statute_ref'))
                        logging.debug(thing.get('parent'))
                        if 'parent' in thing:
                            parent=thing['parent']
                    assert parent is not None
                    logging.debug('look for parent END')
                    dependencies=self.get_children(\
                            {'statute_ref': parent},\
                            self.data_loader.data['structure'])
                    dependencies=self.inverse_polish_notation(\
                            dependencies)
                    stacks[stacki][position]=self.do_operation(operation,\
                        stacks[stacki][position-arity:position],dependencies)
                    # delete operation and inputs to it
                    del stacks[stacki][position-arity:position]
                    del depth_tracker[stacki][position-arity:position]
                    positions[stacki]+=1-arity
                elif states[stacki]==3:
                    # if it's a terminal rule or if the depth budget
                    # is exceeded, resolve with the model
                    batched_operations.append((stacki,position))
                elif states[stacki]==4: # take into account dependencies
                    dependencies=self.get_children(stacks[stacki][position],\
                        self.data_loader.data['structure'])
                    add_stack=self.inverse_polish_notation(dependencies)
                    self.add_data(stacks[stacki][position],add_stack)
                    for x in add_stack:
                        if isinstance(x,tuple): # means it's an operator
                            continue
                        assert 'parent' in x, str(x)
                        assert x['parent']==stacks[stacki][position]['statute_ref']
                    stacks[stacki]=stacks[stacki][:position]+add_stack+[('LOAD',1)]\
                            +stacks[stacki][position:]
                    current_depth=depth_tracker[stacki][position]
                    depth_tracker[stacki]=depth_tracker[stacki][:position]\
                            +[current_depth+1 for _ in add_stack]\
                            +[current_depth]+depth_tracker[stacki][position:]
            # end for stacki,stack in enumerate(stacks)
            # forward a batch through model
            comp_batch=[None for _ in batched_operations]
            for ii,(stacki,position) in enumerate(batched_operations):
                comp_batch[ii]=stacks[stacki][position]
            logging.debug('size of computational batch: ' + str(len(comp_batch)))
            logging.debug('before comp step'.upper())
            logging.debug(comp_batch)
            if len(comp_batch)>0:
                comp_batch=self.to_batch(comp_batch)
                self.forward_single(comp_batch)
                comp_batch=self.from_batch(comp_batch)
            logging.debug('after comp step'.upper())
            logging.debug(comp_batch)
            for ii,(stacki,position) in enumerate(batched_operations):
                stacks[stacki][position]=comp_batch[ii]
            # operations after forwarding through model
            for stacki,stack in enumerate(stacks):
                position=positions[stacki]
                if states[stacki]==0:
                    del stacks[stacki][position-1:position+1]
                    del depth_tracker[stacki][position-1:position+1]
                elif states[stacki]==3:
                    positions[stacki]+=1
            # end for stacki,stack in enumerate(stacks)
        # end while min(len(s) for s in stacks)>0:
        # some relevant predictions might be in the final item's inputs
        # so transfer them to the predictions
        output=[ stack[-1] for stack in stacks ]
        logging.debug('around output'.upper())
        logging.debug(output[0]['logits'].keys())
        stack_sketch=[x['statute_ref'] if isinstance(x,dict) else x
                for x in stack]
        logging.debug('final stack: ' + str(stack_sketch))
        logging.debug('stack output: ' + str(output))
        output=self.to_batch(output)
        return output

    def forward(self,batch,grad=False):
        if self.training_stage<2:
            self.forward_single(batch,grad)
        else:
            batch=self.forward_stack(batch)
        return batch

    def create_span_target(self,context,arg_value):
        tokens=['[CLS]']+self.tokenizer.tokenize(context)+['[SEP]']
        tokenized_value=self.tokenizer.tokenize(arg_value)
        spans=set()
        for start in range(len(tokens)):
            for stop in range(start,len(tokens)):
                if tokens[start:stop+1]==tokenized_value:
                    spans.add((start,stop))
        if len(spans)==0:
            logging.debug('target not found')
            spans.add((0,0))
        assert len(spans)>0, context + ' VS ' + str(arg_value)
        return sorted(spans), len(tokens)

    def loss(self,batch):
        loss_functions=(torch.nn.BCEWithLogitsLoss(reduction='none'), \
            torch.nn.CrossEntropyLoss(reduction='none',ignore_index=-1), \
            torch.nn.KLDivLoss(reduction='sum'))
        loss=[]
        assert len(batch['query_slots'])==len(batch['logits'])
        assert len(batch['query_slots'])==len(batch['context'])
        for queries,logits,context in zip(batch['query_slots'],batch['logits'],batch['context']):
            for arg_name,arg_value in queries:
                if arg_name not in logits:
                    continue
                logit=logits[arg_name]
                argument_type=get_argument_type(arg_name,arg_value)
                assert argument_type in ['binary', 'integer', 'span']
                if argument_type=='binary':
                    target=torch.ones_like(logit) if arg_value \
                            else torch.zeros_like(logit)
                    loss_=loss_functions[0](logit, target)
                    loss_=torch.squeeze(loss_)
                elif argument_type=='integer':
                    target=torch.tensor(int(arg_value))
                    denom=torch.max(0.1*target,5000.*torch.ones_like(target))
                    diff=torch.abs(logit-target)
                    delta=torch.div(diff,denom)
                    loss_=torch.max(delta-1,torch.zeros_like(delta))
                    loss_=torch.squeeze(loss_)
                elif argument_type=='span':
                    try:
                        spans,length=self.create_span_target(context,str(arg_value))
                        start_logit,stop_logit=logit
                        span_logits=[start_logit[a]+stop_logit[b] for a,b in spans]
                        loss_num=torch.logsumexp(torch.stack(span_logits,dim=0),dim=0)
                        allowed_spans=[]
                        for a in range(length):
                            for b in range(length):
                                if a<=b:
                                    allowed_spans.append((a,b))
                        allowed_spans_logits=[start_logit[a]+stop_logit[b] \
                                for a,b in allowed_spans]
                        loss_denom=torch.logsumexp(torch.stack(allowed_spans_logits,dim=0),dim=0)
                        loss_=loss_denom-loss_num # loss is negative logprob
                        logging.debug('span loss ' + str(loss_))
                    except:
                        logging.debug('skip span loss')
                        loss_=None
                if loss_ is not None:
                    loss.append(loss_)
        if len(loss)>0:
            loss=sum(loss)
        else:
            loss=None
        return loss
