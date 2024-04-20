import torch.nn as nn
from torchvision import models


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # Encoder
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Decoder
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        pool1 = self.features_block1(x)
        pool2 = self.features_block2(pool1)
        pool3 = self.features_block3(pool2)
        pool4 = self.features_block4(pool3)
        pool5 = self.features_block5(pool4)

        # Classifier
        score = self.classifier(pool5)

        # Decoder
        pool3_score = self.score_pool3(pool3)
        pool4_score = self.score_pool4(pool4)
        
        # Upsampling
        upscale_2 = self.upscore2(score)

        # Crop upscale_2 to match pool4_score size
        upscale_2_cropped = upscale_2[:, :, :pool4_score.size(2), :pool4_score.size(3)]
        upscale_4 = self.upscore_pool4(upscale_2_cropped + pool4_score)

        # Crop upscale_4 to match pool3_score size
        upscale_4_cropped = upscale_4[:, :, :pool3_score.size(2), :pool3_score.size(3)]
        upscore_final = self.upscore_final(upscale_4_cropped + pool3_score)

        return upscore_final
