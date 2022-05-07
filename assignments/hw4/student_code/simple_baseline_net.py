import torch
import torch.nn as nn
from external.googlenet.googlenet import googlenet


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, question_word_list_length, answer_list_length): # 2.2 TODO: add arguments needed
        super().__init__()
	    ############ 2.2 TODO
        self.image_feature_extractor = googlenet(pretrained=True,
                                                 transform_input=False,
                                                 only_feature_extractor=True)

        # Freezing googlenet.
        for param in self.image_feature_extractor.parameters():
            param.requires_grad = False

        self.word_feature_extractor = nn.Linear(question_word_list_length, 1024)
        self.classifier = nn.Linear(1024 + 1024, answer_list_length)
	    ############

    def forward(self, image, question_encoding):
	    ############ 2.2 TODO

        # print(image.shape)
        # print(question_encoding.shape)

        bow_ques = torch.sum(question_encoding,
                             dim=1).view(-1, question_encoding.size(-1))
        img_feat = self.image_feature_extractor(image).view(-1, 1024)
        word_feat = self.word_feature_extractor(bow_ques).view(-1, 1024)
        word_feat = torch.clip(word_feat, min=-1500, max=1500.)
        comb_feat = torch.cat((img_feat, word_feat), dim=-1)

        out = self.classifier(comb_feat)
        out = torch.clip(out, min=-20, max=20)
        # out = nn.functional.softmax(out)

        del img_feat, word_feat, comb_feat

        return out

	    ############
