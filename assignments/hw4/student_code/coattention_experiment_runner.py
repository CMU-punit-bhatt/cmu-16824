import os
import torch
import torch.nn as nn

from student_code.coattention_net import CoattentionNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
from torchvision import transforms as T


class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation):

        ############ 3.1 TODO: set up transform
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        ############
        res18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        image_encoder = nn.Sequential(*list(res18.children())[:-2])
        image_encoder.eval()
        for param in image_encoder.parameters():
            param.requires_grad = False

        question_word_list_length = 5746
        answer_list_length = 1000

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   transform=transform,
                                   question_word_list_length=question_word_list_length,
                                   answer_list_length=answer_list_length,
                                   cache_location=os.path.join(cache_location, "tmp_train"),
                                   ############ 3.1 TODO: fill in the arguments
                                   question_word_to_id_map=None,
                                   answer_to_id_map=None,
                                   ############
                                   pre_encoder=image_encoder)
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 transform=transform,
                                 question_word_list_length=question_word_list_length,
                                 answer_list_length=answer_list_length,
                                 cache_location=os.path.join(cache_location, "tmp_val"),
                                 ############ 3.1 TODO: fill in the arguments
                                 question_word_to_id_map=train_dataset.question_word_to_id_map,
                                 answer_to_id_map=train_dataset.answer_to_id_map,
                                 ############
                                 pre_encoder=image_encoder)

        self._model = CoattentionNet(train_dataset.question_word_list_length,
                                     train_dataset.answer_list_length)

        super().__init__(train_dataset, val_dataset, self._model, batch_size, num_epochs,
                         num_data_loader_workers=num_data_loader_workers, log_validation=False)

        ############ 3.4 TODO: set up optimizer
        self._optim = torch.optim.Adam(self._model.parameters(), lr=4e-4)

        ############

    def _optimize(self, predicted_answers, true_answer_ids):
        ############ 3.4 TODO: implement the optimization step

        self._optim.zero_grad()

        loss = torch.nn.CrossEntropyLoss()(predicted_answers, true_answer_ids)
        loss.backward()

        self._optim.step()
        ############
