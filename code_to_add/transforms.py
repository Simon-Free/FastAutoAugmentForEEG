def transform_masking(window, policy_list):
    n_fft = 512
    hop_length = 64
    win_length = n_fft
    spectrogram = torch.stft(window, n_fft=n_fft, 
                                  hop_length=hop_length,
                                  win_length=n_fft,
                                  window=torch.hann_window(n_fft))
    
    for policy in policy_list:
        spectrogram = policy.apply(spectrogram)
        
    to_plot = torch.norm(spectrogram, dim=3)
    to_plot = torchaudio.transforms.AmplitudeToDB().forward(to_plot)
    
    
    tensor_to_img(to_plot)
    augmented_window = torchaudio.functional.istft(spectrogram,    
                                                   n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   win_length=n_fft,
                                                   window=torch.hann_window(n_fft))
    
    return(spectrogram)

