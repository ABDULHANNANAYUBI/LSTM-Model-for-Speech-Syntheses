def gen_parameters(y_predicted):
    # Number of time frames
    T = y_predicted.shape[0]

    # Split acoustic features
    mgc = y_predicted[:,:lf0_start_idx]
    lf0 = y_predicted[:,lf0_start_idx:vuv_start_idx]
    vuv = y_predicted[:,vuv_start_idx]
    bap = y_predicted[:,bap_start_idx:]

    # Perform MLPG
    ty = "acoustic"
    mgc_variances = np.tile(Y_var[ty][:lf0_start_idx], (T, 1))
    mgc = paramgen.mlpg(mgc, mgc_variances, windows)
    lf0_variances = np.tile(Y_var[ty][lf0_start_idx:vuv_start_idx], (T,1))
    lf0 = paramgen.mlpg(lf0, lf0_variances, windows)
    bap_variances = np.tile(Y_var[ty][bap_start_idx:], (T, 1))
    bap = paramgen.mlpg(bap, bap_variances, windows)

    return mgc, lf0, vuv, bap