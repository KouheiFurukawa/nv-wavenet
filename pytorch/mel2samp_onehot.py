# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met: 
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# *****************************************************************************
"""
Generating pairs of mel-spectrograms and original audio
"""
import argparse
import json
import os
import random
import torch
import torch.utils.data
import sys

import utils

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT
sys.path.remove('tacotron2')

sys.path.insert(0, 'cpc_model')
from cpc_model.model import audio_model
sys.path.remove('cpc_model')

class Mel2SampOnehot(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, segment_length, mu_quantization,
                 filter_length, hop_length, win_length, sampling_rate):
        audio_files = utils.files_to_list(training_files)
        self.audio_files = audio_files
        random.seed(1234)
        random.shuffle(self.audio_files)
        
        self.stft = TacotronSTFT(filter_length=filter_length,
                                    hop_length=hop_length,
                                    win_length=win_length,
                                    sampling_rate=sampling_rate,
                                    mel_fmin=0.0, mel_fmax=8000.0)
        
        self.segment_length = segment_length
        self.mu_quantization = mu_quantization
        self.sampling_rate = sampling_rate
        self.enc = audio_model()
        state = torch.load('/data/unagi0/furukawa/cpc_logs/logs/stride_256_dim_128/best_checkpoint.tar')
        self.enc.load_state_dict(state)
        for p in self.enc.parameters():
            p.requires_grad = False

    def get_mel(self, audio):
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec
    
    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = utils.load_wav_to_torch(filename)
        audio = torch.clamp(audio, -1., 1.)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        
            # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        with torch.no_grad():
            mel, _ = self.enc.model.get_latent_representations(audio.view(1, 1, -1).float())
            mel = mel.squeeze().transpose(0, 1)
        audio = utils.mu_law_encode(audio, self.mu_quantization)
        return (mel, audio)
    
    def __len__(self):
        return len(self.audio_files)


if __name__ == "__main__":
    """
    Turns audio files into mel-spectrogram representations for inference

    Uses the data portion of the config for audio processing parameters, 
    but ignores training files and segment lengths.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--audio_list", required=True, type=str,
                        help='File containing list of wavefiles')
    parser.add_argument('-o', "--output_dir", required=True, type=str,
                        help='Directory to put Mel-Spectrogram Tensors')
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    
    args = parser.parse_args()

    filepaths = utils.files_to_list(args.audio_list)
    
    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)
    
    # Parse config.  Only using data processing
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    data_config = config["data_config"]
    mel_factory = Mel2SampOnehot(**data_config)  
    
    for filepath in filepaths:
        audio, sampling_rate = utils.load_wav_to_torch(filepath)
        assert(sampling_rate == mel_factory.sampling_rate)
        melspectrogram = mel_factory.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)

