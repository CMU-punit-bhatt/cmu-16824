import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from external.vqa.vqa import VQA

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        self._question_ids = self._vqa.getQuesIds()

        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        # Create the question map if necessary
        if question_word_to_id_map is None:
            ############ 1.6 TODO

            qids = self._vqa.getQuesIds()
            questions = [self._vqa.qqa[id]['question'] for id in qids]

            words_list = self._create_word_list(questions)
            self.question_word_to_id_map = \
                self._create_id_map(words_list, question_word_list_length)

            ############
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary
        if answer_to_id_map is None:
            ############ 1.7 TODO

            # All answers are just words.
            qids = self._vqa.getQuesIds()
            answers = []

            for id in qids:
                answers.extend([a['answer'] for a in self._vqa.loadQA([id])[0]['answers']])

            self.answer_to_id_map = \
                self._create_id_map(answers, answer_list_length)
            ############
        else:
            self.answer_to_id_map = answer_to_id_map

        # print(len(self.answer_to_id_map.keys()))

        self.id_to_answer_map = {k: v for v, k in self.answer_to_id_map.items()}
        self.id_to_answer_map[self.answer_list_length - 1] = '<Unknown>'
        # print(len(self.id_to_answer_map.keys()))

    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """

        words = []
        punctuations = '?.!/;:,'

        # 1-D words list
        for s in sentences:

            s_words = ''.join(list(filter(lambda c: c not in punctuations,
                                          s.lower()))).split(' ')

            if '' in s_words:
                s_words.remove('')

            words.extend(s_words)

        return words


    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """

        words_arr = np.array(word_list, dtype='U')
        u_words, counts = np.unique(words_arr, return_counts=True)

        words_desc = u_words[np.argsort(counts)[::-1]].tolist()
        # counts_desc = counts[np.argsort(counts)[::-1]].tolist()

        lim = max_list_length if max_list_length < len(words_desc) else len(words_desc)

        rank_map = {k: v for v, k in enumerate(words_desc[: lim])}

        return rank_map


    def __len__(self):
        ############ 1.8 TODO

        return len(self._question_ids)

        ############

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ############ 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API

        qid = self._question_ids[idx]
        img_idx = self._vqa.qqa[qid]['image_id']

        ############

        if self._cache_location is not None and self._pre_encoder is not None:
            # the caching and loading logic here
            feat_path = os.path.join(self._cache_location, f'{img_idx}.pt')
            try:
                image = torch.load(feat_path)
            except:
                image_path = os.path.join(
                    self._image_dir, self.get_img_name(img_idx))
                # print(image_path)
                image = Image.open(image_path).convert('RGB')
                # print(image)
                image = self._transform(image).unsqueeze(0)
                image = self._pre_encoder(image)[0]

                torch.save(image, feat_path)
        else:
            ############ 1.9 TODO
            # load the image from disk, apply self._transform (if not None)
            image_path = os.path.join(
                self._image_dir, self.get_img_name(img_idx))
            image = Image.open(image_path).convert('RGB')

            if self._transform is not None:
                image = self._transform(image)
            else:
                image = T.ToTensor()(image)
            ############

        ############ 1.9 TODO
        # load and encode the question and answers, convert to torch tensors

        question = self._vqa.qqa[qid]['question']
        answers = [a['answer'] for a in self._vqa.loadQA([qid])[0]['answers']]
        question_tensor = torch.zeros((self._max_question_length,
                                       self.question_word_list_length))
        answers_tensor = torch.zeros((10, self.answer_list_length))

        q_words = self._create_word_list([question])

        for i, word in enumerate(q_words):

            if word in self.question_word_to_id_map:
                question_tensor[i, self.question_word_to_id_map[word]] = 1
            else:
                question_tensor[i, -1] = 1

        for i, ans in enumerate(answers):

            if ans in self.answer_to_id_map:
                answers_tensor[i, self.answer_to_id_map[ans]] = 1
            else:
                answers_tensor[i, -1] = 1
        
        # print(image.shape)

        ############
        return {
            'qid': qid,
            'image': image,
            'question': question_tensor,
            'answers': answers_tensor
        }

    def get_img_name(self, idx):

        n = 12
        padded_id = '0' * (n - len(str(idx))) + str(idx)

        return self._image_filename_pattern.format(padded_id)