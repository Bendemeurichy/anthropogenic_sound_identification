"""
PANN CNN14 model architecture for audio tagging.

Based on "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"
Reference: https://github.com/qiuqiangkong/audioset_tagging_cnn

This module provides:
- Cnn14: Pretrained CNN14 backbone from AudioSet
- PlaneClassifierPANN: Binary classifier using CNN14 + custom head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from typing import Optional, List

from .config import ModelConfig


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and pooling."""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weight()
    
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception(f'Incorrect pool_type: {pool_type}')
        
        return x


class AttBlock(nn.Module):
    """Attention block for temporal pooling."""
    
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()
        
        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.cla = nn.Conv1d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()
    
    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)
    
    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla
    
    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class Cnn14(nn.Module):
    """
    CNN14 architecture from PANN.
    
    This is the main pretrained backbone that extracts 2048-dimensional
    embeddings from audio spectrograms.
    
    Args:
        sample_rate: Audio sample rate (default: 32000)
        window_size: STFT window size (default: 1024)
        hop_size: STFT hop size (default: 320)
        mel_bins: Number of mel frequency bins (default: 64)
        fmin: Minimum frequency for mel filterbank (default: 50)
        fmax: Maximum frequency for mel filterbank (default: 14000)
        classes_num: Number of output classes (default: 527 for AudioSet)
    """
    
    def __init__(
        self,
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    ):
        super(Cnn14, self).__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True
        )
        
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True
        )
        
        # Spec augmentation
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2
        )
        
        self.bn0 = nn.BatchNorm2d(64)
        
        # Convolutional blocks
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        # Attention block for temporal pooling
        self.att_block = AttBlock(2048, 2048, activation='linear')
        
        self.init_weight()
    
    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
    
    def forward(self, input, return_embedding=False):
        """
        Forward pass through CNN14.
        
        Args:
            input: (batch_size, data_length) - raw waveform
            return_embedding: If True, return embeddings instead of logits
            
        Returns:
            If return_embedding=False:
                (clipwise_output, embedding) where:
                - clipwise_output: (batch_size, classes_num) - classification logits
                - embedding: (batch_size, 2048) - embedding vector
            If return_embedding=True:
                embedding: (batch_size, 2048) - embedding vector only
        """
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)
        
        # Convolutional blocks
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = torch.mean(x, dim=3)
        
        # Attention temporal pooling
        (x1, _, _) = self.att_block(x)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        
        if return_embedding:
            return embedding
        
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding
        }
        
        return output_dict


class PlaneClassifierPANN(nn.Module):
    """
    Binary classifier for plane/no-plane audio using PANN CNN14 backbone.
    
    Architecture:
        Input: Raw waveform (batch_size, samples)
        ↓
        CNN14 backbone → 2048-dim embedding
        ↓
        Dense(512) + ReLU + BatchNorm + Dropout(0.3)
        ↓
        Dense(256) + ReLU + BatchNorm + Dropout(0.2)
        ↓
        Dense(128) + ReLU + BatchNorm + Dropout(0.1)
        ↓
        Dense(1) → Binary logit
    
    Args:
        cnn14: Pretrained CNN14 model
        config: ModelConfig instance with architecture parameters
        fine_tune: Whether CNN14 backbone is trainable
    """
    
    def __init__(
        self,
        cnn14: Cnn14,
        config: Optional[ModelConfig] = None,
        fine_tune: bool = False
    ):
        super(PlaneClassifierPANN, self).__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.config = config
        self.cnn14 = cnn14
        self._fine_tune = fine_tune
        
        # Freeze/unfreeze CNN14
        self.set_fine_tune(fine_tune)
        
        # Build classification head
        layers = []
        prev_units = config.embedding_dim
        
        for i, (units, dropout) in enumerate(zip(config.hidden_units, config.dropout_rates)):
            # Dense layer
            layers.append(nn.Linear(prev_units, units))
            
            # Activation
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'swish':
                layers.append(nn.SiLU())  # SiLU is same as Swish
            else:
                raise ValueError(f"Unsupported activation: {config.activation}")
            
            # Batch normalization
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(units))
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            prev_units = units
        
        # Output layer (binary classification logit)
        layers.append(nn.Linear(prev_units, 1))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize weights for the classifier head."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                init_layer(module)
            elif isinstance(module, nn.BatchNorm1d):
                init_bn(module)
    
    def set_fine_tune(self, fine_tune: bool):
        """
        Set whether to fine-tune the CNN14 backbone.
        
        Args:
            fine_tune: If True, CNN14 is trainable; if False, frozen
        """
        self._fine_tune = fine_tune
        for param in self.cnn14.parameters():
            param.requires_grad = fine_tune
    
    @property
    def fine_tune(self) -> bool:
        """Get current fine-tune state."""
        return self._fine_tune
    
    @fine_tune.setter
    def fine_tune(self, value: bool):
        """Set fine-tune state and update CNN14 trainability."""
        self.set_fine_tune(value)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input: Raw waveform tensor of shape (batch_size, samples)
            
        Returns:
            Logits for binary classification of shape (batch_size, 1)
            Apply sigmoid to get probabilities: torch.sigmoid(logits)
        """
        # Extract embeddings from CNN14
        embedding = self.cnn14(input, return_embedding=True)
        
        # Classification head
        logits = self.classifier(embedding)
        
        return logits
    
    def get_embedding(self, input: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN14 embeddings without classification.
        
        Args:
            input: Raw waveform tensor of shape (batch_size, samples)
            
        Returns:
            Embeddings of shape (batch_size, 2048)
        """
        return self.cnn14(input, return_embedding=True)
