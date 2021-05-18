import argparse, glob, random, json
import torch
from torch import nn
from transformers import *
import allennlp.modules.conditional_random_field as crf

random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Train BERT-based CRF for BIO tagging')
parser.add_argument('--datafile', type=str,
                    help='path to data file')
parser.add_argument('--dev', type=str, nargs='+',
                    help='sections to use as dev')
parser.add_argument('--test', type=str, nargs='+',
                    help='sections to use as test')
parser.add_argument('--expdir', type=str,
                    help='path to experiment directory')
parser.add_argument('--thaw_top_layer', action='store_true',
                    help="whether to thaw BERT's top layer")
parser.add_argument('--debug', action='store_true',
                    help='whether to set the logging level to debug')
parser.add_argument('--max_epochs', type=int, default=100,
                    help='max number of epochs to train for')
parser.add_argument('--patience', type=int, default=25,
        help='number of epochs to train for beyond last improvement on dev set')
parser.add_argument('--batch', type=int, default=32,
                    help='size of batch')
parser.add_argument('--update_period', type=int, default=32,
                    help='how many batches to process before updating parameters')
parser.add_argument('--max_length', type=int, default=512,
                    help='max length of input string')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--bert_model', type=str, default='bert-base-cased',
                    help='which bert model to use')
args = parser.parse_args()

# set up logging
if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
logging.info('arguments:\n'+str(args))

# occupy gpu
logging.info('using gpu')
a = torch.FloatTensor(1).to('cuda')

def log_outputs(output, batch):
    assert output is not None
    assert batch is not None
    output_file=args.expdir.rstrip('/')+'/test_outputs.json'
    with open(output_file,'a') as f:
        for t,ps,gs,s,cs in zip(batch['tokens'],output['spans'],
                             batch['spans'],batch['subsection'],
                             batch['clusters']):
            ps = ps if isinstance(ps,list) else sorted(ps)
            gs = gs if isinstance(gs,list) else sorted(gs)
            op={'tokens': t, 'predicted_spans': ps,
                'gold_spans': gs, 'subsection': s,
                'clusters': cs}
            f.write(json.dumps(op)+'\n')

class DataLoader:
    def __init__(self,datafile,dev,test,batch_size):
        self.batch_size=batch_size
        self.dev=set(dev)
        self.test=set(test)
        self._data=json.load(open(datafile,'r'))
        self.statute_ids=set(self._data['section'])
    # end def __init__

    def init_batches(self,split):
        assert split in ['train','dev','test']
        self.data={'input': [], 'target': [], 'spans': [],
                   'tokens': [], 'subsection': [], 'clusters': []}
        if split=='dev':
            keys=sorted(self.dev)
        elif split=='test':
            keys=sorted(self.test)
        else:
            keys=set(self.statute_ids)
            keys = keys - self.dev
            keys = keys - self.test
            keys = sorted(keys)
        for ii,_ in enumerate(self._data['input']):
            if self._data['section'][ii] in keys:
                self.data['clusters'].append(self._data['projected_clusters'][ii])
                self.data['subsection'].append(self._data['subsection'][ii])
                self.data['tokens'].append(self._data['tokens'][ii])
                self.data['input'].append(self._data['bert_input'][ii])
                self.data['target'].append(self._data['projected_targets'][ii])
                self.data['spans'].append(set(
                    tuple(x) for x in self._data['projected_spans'][ii] ))
        self.position=0
    # end def init_batches(self,split='train')

    def next_batch(self):
        batch={}
        if self.position>len(self.data['input']):
            return None
        start=self.position
        stop=start+self.batch_size
        batch['clusters']=self.data['clusters'][start:stop]
        batch['subsection']=self.data['subsection'][start:stop]
        batch['tokens']=self.data['tokens'][start:stop]
        batch['input']=self.data['input'][start:stop]
        batch['target']=self.data['target'][start:stop]
        batch['spans']=self.data['spans'][start:stop]
        self.position+=self.batch_size
        assert len(batch['input'])==len(batch['target'])
        assert len(batch['input'])==len(batch['spans'])
        if len(batch['input'])==0: # empty batch, can happen at times
            return None
        return batch
    # end def next_batch(self)

    def print_position(self):
        p=str(self.position)
        l=len(self.data['input'])
        return str(p) + '/' + str(l)
    # end def position(self)

# end class DataLoader

class Model(nn.Module):
    def __init__(self,pretrained_model='bert-base-uncased',
            max_length=512):
        super(Model, self).__init__()
        self.pretrained_model=pretrained_model
        # BERT base
        try:
            self.bert=AutoModel.from_pretrained(pretrained_model)
        except:
            self.bert=BertModel.from_pretrained(pretrained_model)
        # lighten the load of the computation
        self.bert.config.output_hidden_states=False
        self.bert.config.output_attentions=False
        # LSTM with attention over context to predict value
        vocab_size=3
        self.predictor=nn.Linear(self.bert.config.hidden_size,vocab_size)
        bio_transitions=crf.allowed_transitions(constraint_type='BIO', \
                labels={0: 'B', 1: 'I', 2: 'O'}) # 3 is start of sequence, 4 is end of sequence
        self.crf=crf.ConditionalRandomField(num_tags=3,constraints=bio_transitions)
        self.bert_max_length=max_length
        self.gpu=False

    def cuda(self):
        self.gpu=True
        self.bert.to('cuda')
        self.predictor.to('cuda')
        self.crf.to('cuda')

    def freeze_bert(self, thaw_top_layer=False):
        for param in self.bert.parameters():
            param.requires_grad=False
        if thaw_top_layer:
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad=True

    def extract_spans(self,tag_seq):
        spans=set()
        start,end=0,0
        while start<len(tag_seq):
            if tag_seq[start]=='B':
                end=start+1
                while end<len(tag_seq):
                    if tag_seq[end]!='I':
                        break
                    end+=1
                spans.add((start,end-1))
                start=end-1
            start+=1
        return spans

    def forward(self,input_ids,targets=None):
        output={}
        token_type_ids = [ [0]*len(a) for a in input_ids ]
        input_mask = [ [1]*len(a) for a in input_ids]
        # pad
        bert_input_ids = list(map(lambda a: \
                a+[0]*(self.bert_max_length-len(a)) \
                if len(a)<self.bert_max_length else a, input_ids))
        token_type_ids = list(map(lambda a: \
                a+[0]*(self.bert_max_length-len(a)) \
                if len(a)<self.bert_max_length else a, token_type_ids))
        input_mask = list(map(lambda a: \
                a+[0]*(self.bert_max_length-len(a)) \
                if len(a)<self.bert_max_length else a, input_mask))
        # turn into tensor + move to gpu
        input_tensor=torch.LongTensor(bert_input_ids).to('cuda')
        token_type_ids_tensor=torch.LongTensor(token_type_ids).to('cuda')
        mask_tensor=torch.FloatTensor(input_mask).to('cuda')
        assert torch.equal(torch.gt(input_tensor,0).byte(),\
                mask_tensor.byte())
        # run BERT
        representations,_=self.bert(input_tensor,\
                token_type_ids=token_type_ids_tensor, \
                attention_mask=mask_tensor) # batch x len x dim
        # feed through classifier
        logits=self.predictor(representations) # batch x len x 3
        output['logits']=logits
        output['logits_normalized']=output['logits']
        # output prediction (character level)
        output['predictions']=[None for _ in range(logits.size(0))]
        output['spans']=[None for _ in range(logits.size(0))]
        # mark start of sequence and end of sequence as not part of the sequence in mask
        for ii,st in enumerate(input_ids):
            ll=len(st)
            mask_tensor[ii,0]=0.
            mask_tensor[ii,ll-1]=0.
        output['predictions']=self.crf.viterbi_tags(output['logits_normalized'], mask_tensor)
        output['predictions']=[x[0] for x in output['predictions']] # delete viterbi scores
        output['predictions']=[list(map(lambda y: ['B','I','O'][y], x)) \
                for x in output['predictions']] # delete viterbi scores
        for bb in range(logits.size(0)):
            # tag predictions don't contain predictions for [SEP] and [CLS]
            assert len(output['predictions'][bb])==len(input_ids[bb])-2
            output['spans'][bb]=self.extract_spans(output['predictions'][bb])
        return output
# end class Model

def score_with_tolerance(gold,candidate,tolerance):
    # precision
    n_correct=0
    for c in candidate:
        assert isinstance(c,tuple)
        assert len(c)==2
        correct=False
        for start_offset in range(-tolerance,tolerance+1):
            for end_offset in range(-tolerance,tolerance+1):
                adjusted_span=(c[0]+start_offset,c[1]+end_offset)
                correct = correct or (adjusted_span in gold)
        n_correct+=int(correct)
    precision=(n_correct,len(candidate))
    assert n_correct<=len(candidate)
    # recall
    n_correct=0
    for g in gold:
        assert isinstance(g,tuple)
        assert len(g)==2
        correct=False
        for start_offset in range(-tolerance,tolerance+1):
            for end_offset in range(-tolerance,tolerance+1):
                adjusted_span=(g[0]+start_offset,g[1]+end_offset)
                correct = correct or (adjusted_span in candidate)
        n_correct+=int(correct)
    recall=(n_correct,len(gold))
    assert n_correct<=len(gold)
    return precision,recall
# end def score_with_tolerance

# load data
data_loader=DataLoader(args.datafile,args.dev,args.test,args.batch)
# load model
model=Model(args.bert_model,args.max_length)
model.cuda()
model.freeze_bert(args.thaw_top_layer)
loss_function=torch.nn.CrossEntropyLoss(reduction='none',ignore_index=-1)
optimizer=AdamW(model.parameters(),lr=args.learning_rate)
model_savedir=args.expdir.rstrip('/')
best_model_savefile=model_savedir+'/best_model.pt'

# begin define forward pass over data
def forward_pass(model,data_loader,split):
    metrics={'tags_correct': 0, 'num_tags': 0, 'num_correct': 0,
            'num_expected': 0, 'num_returned': 0, 'loss': 0, 'num-precision-5': 0,
            'num-recall-5': 0, 'denom-precision-5': 0, 'denom-recall-5': 0}
    update_counter=0
    if split=='train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    data_loader.init_batches(split)
    batch=data_loader.next_batch()
    while batch is not None:
        logging.debug('position: ' + data_loader.print_position())
        # feed through model
        logging.debug('feed through model')
        if split=='train':
            output=model.forward(batch['input'],
                    targets=batch['target'])
        else:
            with torch.no_grad():
                output=model.forward(batch['input'])
        if split=='test':
            log_outputs(output,batch)
        logging.debug('compute loss')
        # loss
        target=[ [['B','I','O'].index(x) for x in y] for y in batch['target'] ]
        target_mask = [[1]*len(a) for a in target]
        target_mask=[ x+[0]*(args.max_length-1-len(x)) \
                if len(x) < args.max_length-1 else x for x in target_mask ]
        target=[ x+[0]*(args.max_length-1-len(x)) \
                if len(x) < args.max_length-1 else x for x in target ]
        target=torch.LongTensor(target).contiguous().to('cuda')
        target_mask=torch.LongTensor(target_mask).contiguous().to('cuda')
        logits=output['logits_normalized'][:,1:,:].contiguous() # remove [CLS] token at start
        llk=model.crf(logits,target,target_mask)
        loss=-llk
        # compute metrics
        update_counter+=len(batch['input'])
        metrics['num_correct']+=sum( len(set(x) & set(s)) \
                for x,s in zip(output['spans'],batch['spans']) )
        metrics['num_expected']+=sum( len(x) for x in batch['spans'] )
        metrics['num_returned']+=sum( len(x) for x in output['spans'] )
        metrics['num_tags']+=sum( len(x) for x in batch['target'] )
        metrics['tags_correct'] += sum( sum( 1 if _x==_y else 0 for _x,_y in zip(x,y) ) \
                for x,y in zip(output['predictions'], batch['target']) )
        assert metrics['num_correct'] <= metrics['num_expected']
        assert metrics['num_correct'] <= metrics['num_returned']
        for x,s in zip(output['spans'],batch['spans']):
            prec_5,reca_5=score_with_tolerance(s,x,5)
            metrics['num-precision-5']+=prec_5[0]
            metrics['denom-precision-5']+=prec_5[1]
            metrics['num-recall-5']+=reca_5[0]
            metrics['denom-recall-5']+=reca_5[1]
        assert metrics['num-precision-5'] <= metrics['denom-precision-5']
        assert metrics['num-recall-5'] <= metrics['denom-recall-5']
        metrics['loss']+=sum(loss.cpu().tolist())
        logging.debug('\t'.join(\
                [str(k)+': '+str(v) for k,v in metrics.items()]))
        # compute backward pass
        if split=='train':
            logging.debug('compute backward pass')
            loss=torch.sum(loss)
            loss.backward()
            if update_counter>=args.update_period:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad /= update_counter
                optimizer.step()
                optimizer.zero_grad()
                update_counter=0
        batch=data_loader.next_batch()
    # end while batch is not None
    if split=='train': # update with remaining gradients
        if update_counter>0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= update_counter
            optimizer.step()
            optimizer.zero_grad()
    metrics['recall']=float(metrics['num_correct'])/max(1,metrics['num_expected'])
    metrics['precision']=float(metrics['num_correct'])/max(1,metrics['num_returned'])
    metrics['f1']=0
    if (metrics['recall']>0) and (metrics['precision']>0):
        metrics['f1']=2*metrics['recall']*metrics['precision']\
                /(metrics['recall']+metrics['precision'])
    metrics['recall-5']=float(metrics['num-recall-5'])/max(1,metrics['denom-recall-5'])
    metrics['precision-5']=float(metrics['num-precision-5'])/max(1,metrics['denom-precision-5'])
    metrics['f1-5']=0
    if (metrics['recall-5']>0) and (metrics['precision-5']>0):
        metrics['f1-5']=2*metrics['recall-5']*metrics['precision-5']\
                /(metrics['recall-5']+metrics['precision-5'])
    metrics['average_loss']=float(metrics['loss'])/max(1,metrics['num_tags'])
    metrics['accuracy']=float(metrics['tags_correct'])/max(1,metrics['num_tags'])
    return metrics
# end define forward pass over data
logging.debug('start train')
best_model_metrics=None
patience_counter=0
for ep in range(args.max_epochs):
    logging.info('epoch ' + str(ep) + '/' + str(args.max_epochs))
    logging.info('train')
    train_metrics=forward_pass(model,data_loader,'train')
    logging.debug('train metrics: ' + str(train_metrics))
    logging.info('dev')
    dev_metrics=forward_pass(model,data_loader,'dev')
    logging.debug('dev metrics: ' + str(dev_metrics))
    logging_message='metrics epoch ' + str(ep) \
            + ': train loss ' \
            + "{0:.5f}".format(train_metrics['average_loss']) \
            + ';' + ' dev loss ' \
            + "{0:.5f}".format(dev_metrics['average_loss']) \
            + ';' + ' train recall ' \
            + "{0:.5f}".format(train_metrics['recall']) \
            + ';' + ' dev recall ' \
            + "{0:.5f}".format(dev_metrics['recall']) \
            + ';' + ' train precision ' \
            + "{0:.5f}".format(train_metrics['precision']) \
            + ';' + ' dev precision ' \
            + "{0:.5f}".format(dev_metrics['precision']) \
            + ';' + ' train f1 ' \
            + "{0:.5f}".format(train_metrics['f1']) \
            + ';' + ' dev f1 ' \
            + "{0:.5f}".format(dev_metrics['f1']) \
            + ';' + ' train recall-5 ' \
            + "{0:.5f}".format(train_metrics['recall-5']) \
            + ';' + ' dev recall-5 ' \
            + "{0:.5f}".format(dev_metrics['recall-5']) \
            + ';' + ' train precision-5 ' \
            + "{0:.5f}".format(train_metrics['precision-5']) \
            + ';' + ' dev precision-5 ' \
            + "{0:.5f}".format(dev_metrics['precision-5']) \
            + ';' + ' train f1-5 ' \
            + "{0:.5f}".format(train_metrics['f1-5']) \
            + ';' + ' dev f1-5 ' \
            + "{0:.5f}".format(dev_metrics['f1-5']) \
            + ';' + ' train accuracy ' \
            + "{0:.5f}".format(train_metrics['accuracy']) \
            + ';' + ' dev accuracy ' \
            + "{0:.5f}".format(dev_metrics['accuracy'])
    logging.info(logging_message)
    def comparison_criterion(metrics):
        return (metrics['f1'], metrics['accuracy'], -metrics['average_loss'])
    # save model
    save_model=False # whether to save as best model
    if best_model_metrics is None:
        best_model_metrics=dict(dev_metrics)
        save_model=True
    if comparison_criterion(best_model_metrics) < comparison_criterion(dev_metrics):
        best_model_metrics=dict(dev_metrics)
        save_model=True
    if save_model:
        # save model
        logger.info('save model to ' + best_model_savefile)
        torch.save(model.state_dict(), best_model_savefile)
        with open(model_savedir+'/best_model_info','w') as f:
            f.write(logging_message+'\n')
    # deal with patience
    if save_model:
        patience_counter=0
    else:
        patience_counter+=1
    logging.info('patience counter: ' + str(patience_counter) \
            + '; patience : ' + str(args.patience))
    if args.patience is not None:
        if patience_counter>args.patience:
            logging.info('patience exhausted')
            break
# end for ep in range(args.max_epochs)

# test
logging.info('test')
model.load_state_dict(torch.load(best_model_savefile))
test_metrics=forward_pass(model,data_loader,'test')
logging.info('best_model metrics: ' + str(test_metrics))
logging_message='metrics test best model ' \
        + ': loss ' \
        + "{0:.5f}".format(test_metrics['average_loss']) \
        + ';' + ' recall ' \
        + "{0:.5f}".format(test_metrics['recall']) \
        + ';' + ' precision ' \
        + "{0:.5f}".format(test_metrics['precision']) \
        + ';' + ' f1 ' \
        + "{0:.5f}".format(test_metrics['f1']) \
        + ';' + ' recall-5 ' \
        + "{0:.5f}".format(test_metrics['recall-5']) \
        + ';' + ' precision-5 ' \
        + "{0:.5f}".format(test_metrics['precision-5']) \
        + ';' + ' f1-5 ' \
        + "{0:.5f}".format(test_metrics['f1-5'])
logging.info(logging_message)
with open(model_savedir+'/best_model_test_results','w') as f:
    f.write(logging_message+'\n')
