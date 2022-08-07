import matplotlib.pyplot as plt
def vis_utterance(X, Y, lengths, idx):
    """Visualize the following features:

    1. Linguistic features
    2. Spectrogram
    3. F0
    4. Aperiodicity
    """
    x = X[idx][:lengths[idx]]
    y = Y[idx][:lengths[idx]]

    figure(figsize=(16,20))
    subplot(4,1,1)
    # haha, better than text?
    librosa.display.specshow(x.T, sr=fs, hop_length=hop_length, x_axis="time")

    subplot(4,1,2)
    logsp = np.log(pysptk.mc2sp(y[:,mgc_start_idx:mgc_dim//len(windows)], alpha=alpha, fftlen=fftlen))
    librosa.display.specshow(logsp.T, sr=fs, hop_length=hop_length, x_axis="time", y_axis="linear")

    subplot(4,1,3)
    lf0 = y[:,mgc_start_idx]
    vuv = y[:,vuv_start_idx]
    plot(lf0, linewidth=2, label="Continuous log-f0")
    plot(vuv, linewidth=2, label="Voiced/unvoiced flag")
    legend(prop={"size": 14}, loc="upper right")

    subplot(4,1,4)
    bap = y[:,bap_start_idx:bap_start_idx+bap_dim//len(windows)]
    bap = np.ascontiguousarray(bap).astype(np.float64)
    aperiodicity = pyworld.decode_aperiodicity(bap, fs, fftlen)
    librosa.display.specshow(aperiodicity.T, sr=fs, hop_length=hop_length, x_axis="time", y_axis="linear")
    # colorbar()