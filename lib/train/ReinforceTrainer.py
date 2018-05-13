import datetime
import math
import os
import time

from torch.autograd import Variable
import torch

import lib
import numpy as np
from termcolor import colored

class ReinforceTrainer(object):

    def __init__(self, actor, critic, train_data, eval_data, test_data, 
                    metrics, dicts, optim, critic_optim, opt,
                    trainprocess_logger, stat_logger, samples_logger):
        self.actor = actor
        self.critic = critic
        self.trpro_logger = trainprocess_logger
        self.log = self.trpro_logger.log_print
        self.stat_logger = stat_logger
        self.samples_logger = samples_logger

        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.evaluator = lib.Evaluator(actor, metrics, 
                            dicts, opt, logger=None)

        self.actor_loss_func = metrics["nmt_loss"]
        self.critic_loss_func = metrics["critic_loss"]
        self.sent_reward_func = metrics["sent_reward"]

        self.optim = optim
        self.critic_optim = critic_optim

        self.max_length = opt.max_predict_length
        self.dicts = dicts
        self.opt = opt

        self.curriculum = range(1, opt.max_predict_length + 1)
        self.update = True

        self.log(actor)
        self.log(critic)

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        self.optim.last_loss = self.critic_optim.last_loss = None
        self.optim.set_lr(self.opt.reinforce_lr)
        self.critic_optim.set_lr(self.opt.reinforce_lr)

        for epoch in range(start_epoch, end_epoch + 1):
            self.log("""Actor optim lr: {:g}; Critic optim lr: {:g}"""\
                .format(self.optim.lr, self.critic_optim.lr))

            if self.opt.use_bipnmt:
                self.log("BIP-NMT Training")
                train_reward, critic_loss, total_requests, total_sents = \
                    self.train_epoch_bipnmt(epoch)
            else:
                self.log('* Sentence-Level Reinforce Training')
                train_reward, critic_loss, total_requests, total_sents = \
                    self.train_epoch(epoch)

            epoch_summary = """Epoch {} End! \
                            \nTrain sentence reward: {:.4g} \
                            \nCritic loss: {:.4g} \
                            \nEvaluation against validation data:"""\
                            .format(epoch, train_reward * 100, critic_loss)
            
            self.log(epoch_summary, to_console=True)
            valid_sent_reward = self.logValidInfo(None, total_requests, 
                                    total_sents, self.eval_data,
                                    return_valid_reward=True)
            self.log("Evaluation against Test Data")
            self.logValidInfo(None, None, None, self.test_data)
            self.save(epoch)


    def train_epoch_bipnmt(self, epoch):
        self.actor.train()
        self.critic.train()
        self.train_data.shuffle()

        total_reward, report_reward = 0, 0
        total_critic_loss, report_critic_loss = 0, 0
        total_words, report_words = 0, 0
        total_requests, report_requests = 0, 0
        total_sents = 0
        total_num_words = 0 
        cum_entropy = 0.

        for i in range(len(self.train_data)):
            batch = self.train_data[i]
            sources = batch[0]
            targets = batch[1]
            self.log_corpus((sources, targets))
            batch_size = targets.size(1)

            log_condition = (i % self.opt.log_interval == 0)
            baseline = 0
            self.buffer_count = 0
            self.replay = self.replay_prefix = self.replay_reward = None

            attention_mask = sources[0].data.eq(lib.Constants.PAD).t()
            self.actor.decoder.attn.applyMask(attention_mask)

            for idx, l in enumerate(self.curriculum): 
                self.actor.zero_grad()
                self.critic.zero_grad()

                # Sample a partial translation 
                samples, outputs, break_signal, avg_entropy, keep_entropy, keep_prob = \
                    self.actor.sample(batch, l, return_entropy=True,
                        prefix=self.replay_prefix, return_prob=True)
                end_cond = break_signal or (idx + 1) == len(self.curriculum)

                ## Output entropy to external file for the entropy curve
                cum_entropy = self.log_entropy(cum_entropy, keep_entropy, total_num_words)                

                # Ask for feedback or not.
                intbl_prev, baseline, avg_H = self.get_UpdateSignal(baseline, idx, avg_entropy, end_cond)

                # feedback is used for update only if self.update is False
                phlv = True if not end_cond else False
                rewards, samples, samples_words, ref_words = \
                    self.sent_reward_func(samples.t().tolist(), 
                        targets.data.t().tolist(), phraselevel=phlv, 
                        return_samples=True)
                orig_reward = sum(rewards) 
                tuned_rewards = lib.Reward.refine_reward(orig_reward, 
                                    self.replay_reward, pred=samples[0], 
                                    replay=self.replay)

                samples = Variable(torch.LongTensor(samples).t().contiguous()).cuda()
                if self.replay is not None:
                    rewards = Variable(torch.FloatTensor([tuned_rewards]).t().contiguous()).cuda()
                else:
                    rewards = Variable(torch.FloatTensor([rewards] * samples.size(0)).contiguous()).cuda()

                # Update Critic 
                to_actor, stats = self.update_critic(sources, samples, rewards)
                critic_weights, baselines = to_actor
                num_words, critic_loss = stats

                # Update actor
                update_buffer = (orig_reward >= self.opt.mu)
                actor_weights = self.update_actor((outputs, samples),
                                    (rewards, *to_actor),
                                    update_buffer, orig_reward)

                # log samples
                wordlevel_info = (baselines, rewards, actor_weights, keep_prob)
                sentlevel_info = (self.buffer_count, avg_H, intbl_prev, orig_reward)
                self.log_samples(wordlevel_info, sentlevel_info, samples_words)

                # Gather stats
                if self.update:
                    total_requests += 1
                    total_num_words += num_words
                    total_critic_loss += critic_loss
                    total_reward += orig_reward
                        
                    report_requests += 1
                    report_reward += orig_reward
                    report_critic_loss += critic_loss
                    report_words += num_words

                if end_cond:
                    self.samples_logger.log_print('\n' + '\n', to_console=False)
                    break

            total_sents += 1

            if (log_condition and i > 0):
                self.log("""Epoch %3d, %6d/%d batches;
                    actor reward: %.4f; critic loss: %f; %s elapsed""" %
                    (epoch, i+1, len(self.train_data),
                    (report_reward / report_requests) * 100,
                    report_critic_loss / report_words,
                    str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))
                report_reward = report_requests = report_critic_loss = report_words = 0
                self.logValidInfo(None, total_requests, total_sents, self.eval_data)

        return total_reward / total_requests, total_critic_loss / total_num_words, total_requests, total_sents


    def train_epoch(self, epoch):
        self.actor.train()
        self.critic.train()
        self.train_data.shuffle()

        total_reward, report_reward = 0, 0
        total_critic_loss, report_critic_loss = 0, 0
        total_sents, report_sents = 0, 0
        total_words, report_words = 0, 0
        total_requests = 0

        for i in range(len(self.train_data)):
            batch = self.train_data[i]
            sources = batch[0]
            targets = batch[1]
            batch_size = targets.size(1)
            log_condition = (i % self.opt.log_interval == 0)

            self.actor.zero_grad()
            self.critic.zero_grad()

            # Sample translations
            attention_mask = sources[0].data.eq(lib.Constants.PAD).t()
            self.actor.decoder.attn.applyMask(attention_mask)
            samples, outputs, _, _, _, _ = self.actor.sample(batch, self.max_length, return_prob=False)

            # Calculate rewards
            rewards, samples = self.sent_reward_func(samples.t().tolist(), 
                                    targets.data.t().tolist(), 
                                    phraselevel=False, return_samples=False)
            reward = sum(rewards)

            samples = Variable(torch.LongTensor(samples).t().contiguous()).cuda()
            rewards = Variable(torch.FloatTensor([rewards] * samples.size(0)).contiguous()).cuda()

            # Update critic.
            assert(self.update)
            to_actor, stats = self.update_critic(sources, samples, rewards)
            num_words, critic_loss = stats
            total_requests += 1

            # Update actor
            actor_weights = self.update_actor((outputs, samples),
                                (rewards, *to_actor), False)
            # Gather stats
            total_reward += reward
            report_reward += reward
            total_sents += batch_size
            report_sents += batch_size
            total_critic_loss += critic_loss
            report_critic_loss += critic_loss
            total_words += num_words
            report_words += num_words

            if log_condition and i > 0:
                self.log("""Epoch %3d, %6d/%d batches;
                      actor reward: %.4f; critic loss: %f; %s elapsed""" %
                      (epoch, i+1, len(self.train_data),
                      (report_reward / report_sents) * 100,
                      report_critic_loss / report_words,
                      str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))
                report_reward = report_sents = report_critic_loss = report_words = 0
                self.logValidInfo(None, total_requests, total_sents, self.eval_data)

        return total_reward / total_sents, total_critic_loss / total_words, total_requests, total_sents


    def logValidInfo(self, batch, total_requests, total_sents, data_totest, return_valid_reward=False):
        loss, sent_reward, corpus_reward = self.evaluator.eval(data_totest)
        if batch is not None:
            self.log("Batch : {}/{}".format(batch, len(self.train_data)))

        if ((total_sents is not None) and (total_requests is not None)):
            assert(total_sents != 0)
            self.log("Avg nos. of requests/sent: {:.4g}".\
                format(total_requests / total_sents))
            
        self.log("Validation: Avg. chrF: {:2f}; Corpus BLEU: {:2f}\n".\
            format(sent_reward * 100, corpus_reward * 100))
        if return_valid_reward:
            return(sent_reward)


    def save(self, epoch):
        checkpoint = {
            "model": self.actor, "critic": self.critic,
            "dicts": self.dicts, "opt": self.opt,
            "epoch": epoch, "optim": self.optim,
            "critic_optim": self.critic_optim
        }
        model_name = os.path.join(self.opt.save_dir, "model_%d" % epoch)
        if self.opt.use_bipnmt:
            model_name += "_bipnmt"
        else:
            model_name += "_reinforce"
        model_name += ".pt"
        torch.save(checkpoint, model_name)
        self.log("Save model as %s" % model_name)


    def update_critic(self, src, samp, rewards):
        critic_weights = samp.ne(lib.Constants.PAD).float()
        num_words = critic_weights.data.sum()
        baselines = self.critic((src, samp), eval=False, regression=True)
        if self.update:
            critic_loss = self.critic.backward(baselines, rewards, 
                            critic_weights, num_words, 
                            self.critic_loss_func, regression=True)
            self.critic_optim.step()
        else:
            critic_loss = 0

        output_for_actor = (critic_weights, baselines)
        output_for_stat = (num_words, critic_loss)
        return(output_for_actor, output_for_stat)


    def update_actor(self, actor_inputs, weights, update_buffer=False,
            orig_reward=-1):

        outputs, samples = actor_inputs
        rewards, critic_weights, baselines = weights
        
        norm_rewards = Variable((rewards - baselines).data)
        actor_weights = norm_rewards * critic_weights

        if self.update:
            actor_loss = self.actor.backward(outputs, samples, 
                            actor_weights, 1, self.actor_loss_func, 
                            regression=False)
            self.optim.step()

        if self.update and update_buffer:
            assert(orig_reward >= 0 and self.buffer_count >= 0)
            self.replay = samples.data.t().tolist()[0]
            self.replay_prefix = samples[:, 0].data
            self.replay_reward = orig_reward
            self.buffer_count += 1
        return(actor_weights)


    def get_UpdateSignal(self, baseline, idx, avg_entropy, end_cond):
        intbl_prev = baseline
        avg_H = float(avg_entropy.data.cpu()) 
        # idx + 1 because the enumerate() starts from idx = 0 
        update = True if ((avg_H - baseline) >= self.opt.eps * baseline) else False
        updated_baseline = baseline + (1.0 / (idx + 1)) * (avg_H - baseline)
        self.update= True if end_cond else update
        return(intbl_prev, updated_baseline, avg_H)


    def log_entropy(self, cum_entropy, keep_entropy, total_token_count):
        '''write entropy to txt
           modify the keep_entropy from sample() by cum_entropy
           to compute the cumulative entropy per token generated
        '''
        keep_entropy[0] += cum_entropy
        seq_cum_entropy = np.cumsum(keep_entropy)
        seq_token_count = np.arange(total_token_count+1, total_token_count + len(seq_cum_entropy)+1)
        seq_entropy_per_token = (seq_cum_entropy / seq_token_count).tolist()
        seq_entropy_per_token = ['{:.6g}'.format(e) for e in seq_entropy_per_token]
        [self.stat_logger.log_print(stat, to_console=False) for stat in seq_entropy_per_token]
        cum_entropy = seq_cum_entropy[-1]
        return(cum_entropy)


    def log_corpus(self, parallel_corpus):
        # write the source and ref to .txt
        src, tgt = parallel_corpus
        src_words = [self.dicts['src'].getLabel(wi) for wi in src[0].data.t().tolist()[0]]
        tgt_words = [self.dicts['tgt'].getLabel(wi) for wi in tgt.data.t().tolist()[0]]
        src_ref_message = '{} {} \
                          \n{} {}'.\
                          format('Source   :', ' '.join(src_words),
                          'Reference:', ' '.join(tgt_words))
        self.samples_logger.log_print(src_ref_message, to_console=False)


    def log_samples(self, wordlv_info, sentlv_info, samplesIntoken):
        bl, re, adv, samp_prob = wordlv_info
        buffer_count, avg_H, intbl_prev, orig_reward = sentlv_info

        def format_string(stat):
            try:
                stat_list = stat.data.squeeze().tolist()
            except AttributeError:
                stat_list = stat
            return(' '.join(['{:.4g}'.format(obj) for obj in stat_list]))

        baselines_str = format_string(bl)
        wordlevel_reward_str = format_string(re)
        advantage_str = format_string(adv)
        samples_prob_str = format_string(samp_prob)
        samples_str = ' '.join(samplesIntoken[0])
    
        message = """Request: {} ##BufferCount: {} ##H(p): {:.4g} ##H(p)BaseLine: {:.4g} ##OrigReward: {:.4g}\
                    \nWordlv Reward  : {}\
                    \nReward BaseLine: {}\
                    \nAdvantage      : {}\
                    \nSamples        : {} \
                    \nSp_prob        : {} \n""".\
                    format(self.update, buffer_count, avg_H, 
                        intbl_prev, orig_reward, wordlevel_reward_str, 
                        baselines_str, advantage_str,
                        samples_str, samples_prob_str)

        self.samples_logger.log_print(message, to_console=False)


