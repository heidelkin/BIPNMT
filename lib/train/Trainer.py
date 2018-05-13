import datetime
import math
import os
import time

import torch

import lib

class Trainer(object):
    def __init__(self, model, train_data, eval_data, metrics, dicts,
        optim, opt, trainprocess_logger):

        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.evaluator = lib.Evaluator(model, metrics, 
                            dicts, opt, logger=None)
        self.loss_func = metrics["nmt_loss"]
        self.dicts = dicts
        self.optim = optim
        self.opt = opt

        self.trpro_logger=trainprocess_logger
        self.log = self.trpro_logger.log_print

        self.log(model)

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        for epoch in range(start_epoch, end_epoch + 1):
            self.log('')

            self.log("* XENT epoch *")
            self.log("Model optim lr: %g" % self.optim.lr)
            train_loss = self.train_epoch(epoch)
            self.log('Train perplexity: %.2f' % math.exp(min(train_loss, 100)))

            valid_loss, valid_sent_reward, valid_corpus_reward = self.evaluator.eval(self.eval_data)
            valid_ppl = math.exp(min(valid_loss, 100))

            self.log('Validation perplexity: %.2f' % valid_ppl)
            self.log('Validation sentence reward: %.2f' % (valid_sent_reward * 100))
            self.log('Validation corpus reward: %.2f' %
                (valid_corpus_reward * 100))

            self.optim.updateLearningRate(valid_loss, epoch)

            checkpoint = {
                'model': self.model,
                'dicts': self.dicts,
                'opt': self.opt,
                'epoch': epoch,
                'optim': self.optim,
            }
            model_name = os.path.join(self.opt.save_dir, "model_%d.pt" % epoch)
            torch.save(checkpoint, model_name)
            self.log("Save model as %s" % model_name)

            
    def train_epoch(self, epoch):
        self.model.train()
        self.train_data.shuffle()

        total_loss, report_loss = 0, 0
        total_words, report_words = 0, 0
        last_time = time.time()

        for i in range(len(self.train_data)):
            batch = self.train_data[i]
            targets = batch[1]

            self.model.zero_grad()
            attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()
            self.model.decoder.attn.applyMask(attention_mask)
            outputs = self.model(batch, eval=False)

            weights = targets.ne(lib.Constants.PAD).float()
            num_words = weights.data.sum()
            loss = self.model.backward(outputs, targets, weights, num_words, self.loss_func)

            self.optim.step()

            report_loss += loss
            total_loss += loss
            total_words += num_words
            report_words += num_words
            if i % self.opt.log_interval == 0 and i > 0:
                self.log("""Epoch %3d, %6d/%d batches;
                      perplexity: %8.2f; %5.0f tokens/s; %s elapsed""" %
                      (epoch, i, len(self.train_data),
                      math.exp(report_loss / report_words),
                      report_words / (time.time() - last_time),
                      str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                report_loss = report_words = 0
                last_time = time.time()

        return total_loss / total_words

