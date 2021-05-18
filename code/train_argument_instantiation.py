import sys, os, argparse, torch, random, logging, math, json, matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
from transformers import *
from sklearn.mixture import GaussianMixture
import numpy as np
from argument_instantiation_models import MixedModel, DataLoaderTrain, DataLoaderPretrain, Metrics, Clustering

# occupy gpu
a = torch.FloatTensor(1).to('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

random.seed(2)
torch.manual_seed(2)

TRUTH_KEY='@TRUTH'

parser = argparse.ArgumentParser(description='Train argument instantiation model')
parser.add_argument('--datafile', type=str,
                    help='path to data file')
parser.add_argument('--split_index', type=int,
                    help='which split to use as a dev set')
parser.add_argument('--expdir', type=str,
                    help='path to experiment directory')
parser.add_argument('--thaw_top_layer', action='store_true',
                    help="whether to thaw BERT's top layer")
parser.add_argument('--debug', action='store_true',
                    help='whether to set the logging level to debug')
parser.add_argument('--max_epochs', type=int, default=None,
                    help='max number of epochs to train for')
parser.add_argument('--patience', type=int, default=None,
        help='number of epochs to train for beyond last improvement on dev set')
parser.add_argument('--batch', type=int, default=4,
                    help='size of batch')
parser.add_argument('--update_period', type=int, default=1,
                    help='how many batches to process before updating parameters')
parser.add_argument('--epoch_size', type=int, default=None,
                    help='number of examples per epoch (if None, will do ordinary training)')
parser.add_argument('--max_length', type=int, default=512,
                    help='max length of input string')
parser.add_argument('--max_depth', type=int, default=5,
                    help='max depth of expansion for statutes')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='weight of L2 regularization')
parser.add_argument('--training_stage', type=int, default=None,
                    help='stage of training to be performed')
parser.add_argument('--gmm_model_file', type=str, default=None,
                    help='file to load gmm model from (if file exists) or to write it to (otherwise)')
parser.add_argument('--bert_model', type=str, default='bert-base-cased',
                    help='which bert model to use')
parser.add_argument('--bert_tokenizer', type=str, default='bert-base-cased',
                    help='which bert tokenizer to use')
args = parser.parse_args()

# set up logging
if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
logging.info('arguments:\n'+str(args))

# load data
logging.info('load data')
with open(args.datafile,'r') as f:
    data=json.load(f)
# load dev and train
if args.epoch_size is None:
    data['dev']=list(filter(lambda x: x[-1]==args.split_index, data['train']))
    data['dev']=list(map(lambda x: x[:-1], data['dev']))
    data['train']=list(filter(lambda x: x[-1]!=args.split_index, data['train']))
    data['train']=list(map(lambda x: x[:-1], data['train']))
    data['train']={'gold': data['train'], 'automatic': []}
    data_loader=DataLoaderTrain(data,args.batch)
else:
    data['train']=data['train']['automatic']
    data['train']=list(map(tuple, data['train']))
    data_loader=DataLoaderPretrain(data,args.batch,epoch_size=args.epoch_size)
# load model
# define saving and loading the model
def save_state(model,optimizer,savefile):
    torch.save({'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        savefile)

def load_state(model,optimizer,savefile):
    checkpoint=torch.load(savefile)
    model.load_state_dict(checkpoint['model_state_dict'])

model_savedir=args.expdir.rstrip('/')
if not os.path.isdir(model_savedir):
    os.mkdir(model_savedir)
model_checkpoint_file=model_savedir+'/checkpoint.pt'
best_model_savefile=model_savedir+'/best_model.pt'
logging.info('create or load GMM')
if os.path.isfile(args.gmm_model_file):
    with open(args.gmm_model_file, 'rb') as f:
        gmm_model = pickle.load(f)
else:
    gmm_model=Clustering(data=data_loader.get_integer_values())
    gmm_model.cluster()
    with open(args.gmm_model_file,'wb') as f:
        pickle.dump(gmm_model, f)
model=MixedModel(pretrained_model=args.bert_model, \
        tokenizer=args.bert_tokenizer, max_length=args.max_length, \
        data_loader=data_loader, max_depth=args.max_depth, \
        gmm_model=gmm_model)
model.freeze_bert(thaw_top_layer=args.thaw_top_layer)
optimizer=AdamW(model.parameters(),lr=args.learning_rate,\
        weight_decay=args.weight_decay)
model_file=None
if os.path.isfile(model_checkpoint_file):
    model_file=model_checkpoint_file
elif os.path.isfile(best_model_savefile):
    model_file=best_model_savefile
if model_file is not None:
    logging.info('load saved model from '+model_file)
    load_state(model,optimizer,model_file)
if args.training_stage is None:
    logging.debug('leaving training stage unchanged')
else:
    logging.debug('setting training stage to ' + str(args.training_stage))
    model.set_training_stage(args.training_stage)

logging.info('model:\n'+str(model))
model.cuda()

# TRAIN
def forward_pass(split):
    metrics=Metrics()
    update_counter=0
    if split=='train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    data_loader.init_batches(split)
    batch=data_loader.next_batch()
    while batch is not None:
        # feed through model
        logging.debug('feed through model')
        if split=='train':
            batch=model.forward(batch,grad=True)
        else:
            with torch.no_grad():
                batch=model.forward(batch,grad=False)
        metrics.accumulate(batch)
        logging.debug('compute loss')
        loss=model.loss(batch)
        if loss is None:
            logging.info('position: ' + str(data_loader.position) \
                + '/' + str(len(data_loader.sample)) + '; loss is None')
        else:
            logging.info('position: ' + str(data_loader.position) \
                + '/' + str(len(data_loader.sample)) + '; loss=' + str(loss.cpu().tolist()))
        # compute backward pass
        if (split=='train') and (loss is not None):
            update_counter+=len(batch['context'])
            logging.debug('compute backward pass')
            loss.backward()
            if update_counter>=args.update_period:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad /= update_counter
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
    return metrics
# end define forward pass over data
# DEV before training
best_model_metrics=None
logging.debug('dev before train')
dev_metrics=forward_pass('dev')
dev_metrics.compute_metrics()
dev_logging_message=dev_metrics.print_metrics()
logging.debug('dev metrics: ' + str(dev_logging_message))
logging_message='dev before training metrics: ' \
        + 'dev metrics: ' + dev_logging_message
logging.info(logging_message)
# save model
logger.info('save model to ' + model_checkpoint_file)
save_state(model,optimizer,model_checkpoint_file)
best_model_metrics=dev_metrics
# save model
logger.info('save model to ' + best_model_savefile)
save_state(model,optimizer,best_model_savefile)
with open(model_savedir+'/best_model_info','w') as f:
    f.write(logging_message+'\n')

logging.debug('start train')
patience_counter=0
for ep in range(args.max_epochs):
    # begin define forward pass over data
    logging.info('epoch ' + str(ep) + '/' + str(args.max_epochs))
    logging.info('train')
    train_metrics=forward_pass('train')
    train_metrics.compute_metrics()
    train_logging_message=train_metrics.print_metrics()
    logging.debug('train metrics: ' + str(train_logging_message))
    logging.info('dev')
    dev_metrics=forward_pass('dev')
    dev_metrics.compute_metrics()
    dev_logging_message=dev_metrics.print_metrics()
    logging.debug('dev metrics: ' + str(dev_logging_message))
    logging_message='metrics epoch ' + str(ep) \
            + '; train metrics: ' + train_logging_message \
            + '; dev metrics: ' + dev_logging_message
    logging.info(logging_message)
    # save model
    logger.info('save model to ' + model_checkpoint_file)
    save_state(model,optimizer,model_checkpoint_file)
    save_model=False # whether to save as best model
    if best_model_metrics is None:
        best_model_metrics=dev_metrics
        save_model=True
    if best_model_metrics.sorting_key()<dev_metrics.sorting_key():
        best_model_metrics=dev_metrics
        save_model=True
    if save_model:
        # save model
        logger.info('save model to ' + best_model_savefile)
        save_state(model,optimizer,best_model_savefile)
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
        if patience_counter==args.patience:
            logging.info('patience exhausted')
            break
# end for ep in range(args.max_epochs)

# run on test data
# begin define forward pass over data
def test_forward_pass(model_file):
    load_state(model,optimizer,model_file)
    model.eval()
    metrics=Metrics()
    optimizer.zero_grad()
    data_loader.init_batches('test')
    batch=data_loader.next_batch()
    saved_outputs={}
    while batch is not None:
        logging.info('position: ' + str(data_loader.position) \
                + '/' + str(len(data_loader.sample)))
        logging.debug('feed through model')
        with torch.no_grad():
            batch=model.forward(batch,grad=False)
        metrics.accumulate(batch)
        logging.debug('compute loss')
        # create gold output
        batch['gold_output'] = [ {} for _ in batch['query_slots'] ]
        for ii,_s in enumerate(batch['query_slots']):
            for _x in _s:
                name,value=_x
                batch['gold_output'][ii][name]=value
        logging.debug('compute predictions')
        # compute metrics
        case_refs=batch['case_ref']
        for c in case_refs:
            assert c not in saved_outputs
            saved_outputs[c]=[]
        truth_target=[ 1 if y[TRUTH_KEY] else 0 \
                for y in batch['gold_output'] ]
        truth_target=torch.FloatTensor(truth_target).to('cuda')
        truth_predictions=[ y[TRUTH_KEY] for y in batch['predictions'] ]
        for ii,c in enumerate(case_refs):
            saved_outputs[c].append('|'.join(\
                    (TRUTH_KEY,str(truth_predictions[ii]),str(truth_target[ii].cpu().tolist())) ))
        slot_predictions=batch['predictions']
        for ii,_ in enumerate(slot_predictions):
            del slot_predictions[ii][TRUTH_KEY]
        for ii,_s in enumerate(batch['query_slots']):
            for name,value in _s:
                if name==TRUTH_KEY:
                    continue
                correct=0
                if name in slot_predictions[ii]:
                    saved_outputs[batch['case_ref'][ii]].append('|'.join(\
                            (str(name),str(slot_predictions[ii][name]),str(value)) ))
        batch=data_loader.next_batch()
    # end while batch is not None:
    return metrics, saved_outputs
# end define forward pass over data
# print results
logging.info('testing best model: ' + str(best_model_savefile))
test_metrics,saved_outputs=test_forward_pass(best_model_savefile)
test_metrics.compute_metrics(compute_confidence_intervals=True)
metrics_message=test_metrics.print_metrics()
logging.info('best model metrics: ' + str(metrics_message))
logging_message='metrics test best model ' + metrics_message
logging.info(logging_message)
with open(model_savedir+'/best_model_test_results','w') as f:
    f.write(logging_message+'\n')
with open(model_savedir+'/best_model_test_outputs','w') as f:
    f.write('\n'.join( k+'\t'+'\t'.join(v) for k,v in saved_outputs.items() ) + '\n')
