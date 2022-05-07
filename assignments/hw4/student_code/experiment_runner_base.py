from torch.utils.data import DataLoader
from student_code.utils import AverageMeter, get_accuracy
import torch
import wandb
from PIL import Image
import os

USE_WANDB_GLOBAL = True

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, log_validation=False):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 1500  # Steps
        self._save_freq = 3000 # Steps

        print(f'Train Dataset: {len(train_dataset)}')
        print(f'Val Dataset: {len(val_dataset)}')

        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation

        if USE_WANDB_GLOBAL:
            wandb.init(project='vlr-hw4')
            wandb.watch(model, log_freq=100)
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")

            wandb.define_metric("val/step")
            wandb.define_metric("val/*", step_metric="val/step")

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self, subset=False):
        ############ 2.8 TODO
        # Should return your validation accuracy
        val_losses = AverageMeter()
        val_accuracies = AverageMeter()

        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            self._model.eval()

            if subset and batch_id > 50:
                break

            image = batch_data['image']
            question = batch_data['question']
            answers = batch_data['answers']

            max_count = torch.argmax(torch.sum(answers, dim=1), dim=-1)
            ground_truth_answer = max_count.view(answers.size(0))

            if self._cuda:
                image = image.cuda()
                question = question.cuda()
                ground_truth_answer = ground_truth_answer.cuda()
            
            with torch.no_grad():
                predicted_answer = self._model(image, question)

                loss = torch.nn.CrossEntropyLoss()(predicted_answer,
                                                  ground_truth_answer)

                val_losses.update(loss.item())
                val_accuracies.update(get_accuracy(predicted_answer,
                                                  ground_truth_answer))

            if batch_id < 2:
                qid = batch_data['qid'][0]

                image_id = self._val_dataset._vqa.qqa[qid.item()]['image_id']
                image_path = os.path.join(self._val_dataset._image_dir, 
                                          self._val_dataset.get_img_name(image_id))
                val_image = Image.open(image_path).convert('RGB')
                
                val_question = self._val_dataset._vqa.qqa[qid.item()]['question']
                val_ans = self._val_dataset.id_to_answer_map[
                    ground_truth_answer[0].item()]
                val_pred = self._val_dataset.id_to_answer_map[
                    torch.argmax(predicted_answer[0]).item()]

                if USE_WANDB_GLOBAL:
                    wandb.log({f'val/image_{batch_id}': \
                               wandb.Image(val_image,
                                           caption=f'Question: {val_question}, GT: {val_ans}, Pred: {val_pred}')})

            del batch_data


        ############

        if USE_WANDB_GLOBAL:
            ############ 2.9 TODO
            # you probably want to plot something here
            wandb.log({'val/loss_avg': val_losses.avg})
            wandb.log({'val/acc_avg': val_accuracies.avg})
            wandb.log({'val/loss_val': val_losses.val})
            wandb.log({'val/acc_val': val_accuracies.val})

            ############

        return val_accuracies.avg

    def train(self):

        train_losses = AverageMeter()
        train_accuracies = AverageMeter()

        best_acc = -1.
        val_step = 0

        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.

                # q_id = batch_data['qid']
                image = batch_data['image']
                question = batch_data['question']
                answers = batch_data['answers']

                # Answers - (N, 10, lim)
                max_count = torch.argmax(torch.sum(answers, dim=1), dim=-1)
                ground_truth_answer = max_count.view(answers.size(0))

                if self._cuda:
                    image = image.cuda()
                    question = question.cuda()
                    ground_truth_answer = ground_truth_answer.cuda()

                predicted_answer = self._model(image, question)

                del image

                ############

                # Optimize the model according to the predictions
                self._optimize(predicted_answer, ground_truth_answer)
                loss = torch.nn.CrossEntropyLoss()(predicted_answer,
                                                   ground_truth_answer)
                train_losses.update(loss.item())
                train_accuracies.update(get_accuracy(predicted_answer,
                                                     ground_truth_answer))

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    ############ 2.9 TODO
                    # you probably want to plot something here)
                    if USE_WANDB_GLOBAL:
                        wandb.log({'train/loss_avg': train_losses.avg})
                        wandb.log({'train/acc_avg': train_accuracies.avg})
                        wandb.log({'train/loss_val': train_losses.val})
                        wandb.log({'train/acc_val': train_accuracies.val})
                        wandb.log({'train/step': current_step})

                    ############
                if (current_step) % self._save_freq == 0:
                    torch.save(self._model.state_dict(),
                               f'./checkpoints/model_{current_step}.pt')
                    
                    if best_acc < train_accuracies.avg:
                        torch.save(self._model.state_dict(),
                                  f'./checkpoints/best_model.pt')
                        best_acc = train_accuracies.avg


                if (current_step) % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate(subset=True)
                    val_step += 1
                    print("Epoch: {} has val accuracy {}".format(epoch,
                                                                 val_accuracy))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    if USE_WANDB_GLOBAL:
                        wandb.log({'val/step': val_step})

                    ############
        
        val_accuracy = self.validate()
        print(f'Final Val Accuracy: {val_accuracy}')
        torch.save(self._model.state_dict(), 
                   f'./checkpoints/model_final.pt')
