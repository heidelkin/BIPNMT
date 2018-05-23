import argparse
import os
import numpy as np
import random
import time

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable

import lib

parser = argparse.ArgumentParser(description="train.py")

## Data options
parser.add_argument("-data", required=True,
                    help="Path to the *-train.pt file from preprocess.py")
parser.add_argument("-save_dir", required=True,
                    help="Directory to save models")
parser.add_argument("-load_from", help="Path to load a pretrained model.")

## Model options
parser.add_argument("-layers", type=int, default=1,
                    help="Number of layers in the LSTM encoder/decoder")
parser.add_argument("-rnn_size", type=int, default=500,
                    help="Size of LSTM hidden states")
parser.add_argument("-word_vec_size", type=int, default=500,
                    help="Size of word embeddings")
parser.add_argument("-input_feed", type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")

## Optimization options
parser.add_argument("-batch_size", type=int, default=64,
                    help="Maximum batch size")
parser.add_argument("-eval_batch_size", type=int, default=64,
                    help="""batch_size for evaluation data 
                    esp use in per-sent training""")
parser.add_argument("-end_epoch", type=int, default=50,
                    help="Epoch to stop training.")
parser.add_argument("-start_epoch", type=int, default=1,
                    help="Epoch to start training.")
parser.add_argument("-param_init", type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument("-optim", default="adam",
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument("-lr", type=float, default=1e-3,
                    help="Initial learning rate")
parser.add_argument("-max_grad_norm", type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument("-learning_rate_decay", type=float, default=0.5,
                    help="""Decay learning rate by this much if (i) perplexity
                    does not decrease on the validation set or (ii) epoch has
                    gone past the start_decay_at_limit""")
parser.add_argument("-start_decay_at", type=int, default=5,
                    help="Start decay after this epoch")

# GPU
parser.add_argument("-gpus", default=[0], nargs="+", type=int,
                    help="Use CUDA")
parser.add_argument("-log_interval", type=int, default=100,
                    help="Print stats at this interval.")
parser.add_argument("-seed", type=int, default=3435,
                     help="Seed for random initialization")

# Critic
parser.add_argument("-start_reinforce", type=int, default=None,
                    help="""Epoch to start reinforcement training.
                    Use -1 to start immediately.""")
parser.add_argument("-reinforce_lr", type=float, default=1e-4,
                    help="""Learning rate for reinforcement training.""")

# Type of Reinforce Update
parser.add_argument("-use_bipnmt", action="store_true", default=False)

# Evaluation
parser.add_argument("-eval", action="store_true", help="Evaluate model only")
parser.add_argument("-max_predict_length", type=int, default=50,
                    help="Maximum length of predictions.")

# Reward Signal
parser.add_argument("-use_charF", action="store_true", help="use charF as reward Signal")


# Others
parser.add_argument("-eps", type=float, default=0.75, help="tolerance in entropy constraint")
parser.add_argument("-mu", type=float, default=0.8, help="threshold for a phrase to be stored in Buffer")

opt = parser.parse_args()

# Setup a logger here for logging the training process
trpro_logger = lib.Logger()
if opt.eval:
    save_file = 'evalInfo.txt'
    stat_logger = None
    samples_logger = None
else:
    # training mode
    save_file = 'trainInfo.txt'
    if opt.use_bipnmt:
        stat_logger = lib.Logger()
        stat_logger.set_log_file(os.path.join(opt.save_dir, "stat.txt"))
        samples_logger = lib.Logger()
        samples_logger.set_log_file(os.path.join(opt.save_dir, "samples.txt"))
    else:
        stat_logger = None
        samples_logger = None

trpro_logger.set_log_file(os.path.join(opt.save_dir, save_file))
log = trpro_logger.log_print
log(opt)

# Set seed
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

opt.cuda = len(opt.gpus)

if opt.save_dir and not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

if torch.cuda.is_available() and not opt.cuda:
    log("WARNING: You have a CUDA device, so you should probably run with -gpus 1")

if opt.cuda:
    cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)

def init(model):
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)

def create_optim(model):
    optim = lib.Optim(
        model.parameters(), opt.optim, opt.lr, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay, start_decay_at=opt.start_decay_at
    )
    return optim

def create_model(model_class, dicts, gen_out_size):
    encoder = lib.Encoder(opt, dicts["src"])
    decoder = lib.Decoder(opt, dicts["tgt"])
    generator = lib.BaseGenerator(nn.Linear(opt.rnn_size, gen_out_size), opt)
    model = model_class(encoder, decoder, generator, opt)
    init(model)
    optim = create_optim(model)
    return model, optim

def create_critic(checkpoint, dicts, opt):
    if opt.load_from is not None and "critic" in checkpoint:
        critic = checkpoint["critic"]
        critic_optim = checkpoint["critic_optim"]
    else:
        critic, critic_optim = create_model(lib.NMTModel, dicts, 1)
    if opt.cuda:
        critic.cuda(opt.gpus[0])
    return critic, critic_optim

def main():
    assert(opt.start_epoch <= opt.end_epoch),'The start epoch should be <= End Epoch'
    log('Loading data from "%s"' % opt.data)
    dataset = torch.load(opt.data)

    supervised_data = lib.Dataset(dataset["train_xe"], opt.batch_size, opt.cuda, eval=False)
    bandit_data = lib.Dataset(dataset["train_pg"], opt.batch_size, opt.cuda, eval=False)

    sup_valid_data = lib.Dataset(dataset["sup_valid"], opt.eval_batch_size, opt.cuda, eval=True)
    bandit_valid_data  = lib.Dataset(dataset["bandit_valid"], opt.eval_batch_size, opt.cuda, eval=True)
    test_data = lib.Dataset(dataset["test"], opt.eval_batch_size, opt.cuda, eval=True)

    dicts = dataset["dicts"]
    log(" * vocabulary size. source = %d; target = %d" %
          (dicts["src"].size(), dicts["tgt"].size()))
    log(" * number of XENT training sentences. %d" %
          len(dataset["train_xe"]["src"]))
    log(" * number of PG training sentences. %d" %
          len(dataset["train_pg"]["src"]))
    log(" * number of bandit valid sentences. %d" %
          len(dataset["bandit_valid"]["src"]))
    log(" * number of  test sentences. %d" %
          len(dataset["test"]["src"]))
    log(" * maximum batch size. %d" % opt.batch_size)
    log("Building model...")

    use_critic = opt.start_reinforce is not None

    if opt.load_from is None:
        model, optim = create_model(lib.NMTModel, dicts, dicts["tgt"].size())
        checkpoint = None
    else:
        log("Loading from checkpoint at %s" % opt.load_from)
        checkpoint = torch.load(opt.load_from)
        model = checkpoint["model"]
        optim = checkpoint["optim"]
        opt.start_epoch = checkpoint["epoch"] + 1

    # GPU.
    if opt.cuda:
        model.cuda(opt.gpus[0])

    # Start reinforce training immediately.
    if (opt.start_reinforce == -1):
        opt.start_decay_at = opt.start_epoch
        opt.start_reinforce = opt.start_epoch

    nParams = sum([p.nelement() for p in model.parameters()])
    log("* number of parameters: %d" % nParams)

    # Metrics.
    metrics = {}
    metrics["nmt_loss"] = lib.Loss.weighted_xent_loss
    metrics["critic_loss"] = lib.Loss.weighted_mse
    log(" Simulated Feedback: charF score\nEvaluation: charF and Corpus BLEU")
    instance_charF = lib.Reward.charFEvaluator(dict_tgt=dicts["tgt"])
    metrics["sent_reward"] = instance_charF.sentence_charF
    metrics["corp_reward"] = lib.Reward.corpus_bleu

    # Evaluate model on heldout dataset.
    if opt.eval:
        evaluator = lib.Evaluator(model, metrics, dicts, opt, trpro_logger)

        # On Bandit test data
        pred_file = opt.load_from.replace(".pt", ".test.pred")
        tgt_file = opt.load_from.replace(".pt", ".test.tgt")
        evaluator.eval(test_data, pred_file)
        evaluator.eval(test_data, pred_file=None, tgt_file=tgt_file)

    else:
        xent_trainer = lib.Trainer(model, supervised_data, sup_valid_data, 
                        metrics, dicts, optim, opt, 
                        trainprocess_logger=trpro_logger)
        if use_critic:
            start_time = time.time()
            # Supervised training: used when running pretrain+bandit together
            xent_trainer.train(opt.start_epoch, opt.start_reinforce - 1, start_time)
            # Actor-Critic
            critic, critic_optim = create_critic(checkpoint, dicts, opt)
            reinforce_trainer = lib.ReinforceTrainer(model, critic, bandit_data, bandit_valid_data,
                    test_data, metrics, dicts, optim, critic_optim, 
                    opt, trainprocess_logger=trpro_logger, stat_logger=stat_logger,
                    samples_logger=samples_logger)
            reinforce_trainer.train(opt.start_reinforce, opt.end_epoch, start_time)
            if opt.use_bipnmt:
                stat_logger.close_file()
                samples_logger.close_file()
        else:
            # Supervised training only.
            xent_trainer.train(opt.start_epoch, opt.end_epoch)

    trpro_logger.close_file()

if __name__ == "__main__":
    main()
