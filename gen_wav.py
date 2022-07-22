def gen_waveform(y_predicted, do_postfilter=False):
    y_predicted = trim_zeros_frames(y_predicted)

    # Generate parameters and split streams
    mgc, lf0, vuv, bap = gen_parameters(y_predicted)

    if do_postfilter:
        mgc = merlin_post_filter(mgc, alpha)

    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), fs, fftlen)
    f0 = lf0.copy()
    f0[vuv < 0.5] = 0
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

    generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64),
                                            spectrogram.astype(np.float64),
                                            aperiodicity.astype(np.float64),
                                            fs, frame_period)