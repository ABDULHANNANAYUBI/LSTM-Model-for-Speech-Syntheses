import tensorflow as tf
def gen_duration(label_path, duration_model):
    # Linguistic features for duration
    hts_labels = hts.load(label_path)
    duration_linguistic_features = fe.linguistic_features(hts_labels,
                                               binary_dict, continuous_dict,
                                               add_frame_features=False,
                                               subphone_features=None).astype(np.float32)

    # Apply normalization
    ty = "duration"
    duration_linguistic_features = minmax_scale(duration_linguistic_features,
                                       X_min[ty], X_max[ty], feature_range=(0.01, 0.99))

    # Apply models
    duration_model = duration_model.gpu()
    duration_model.eval()

    #  Apply model
    x = Variable(torch.from_numpy(duration_linguistic_features)).float()
    try:
        duration_predicted = duration_model(x).data.numpy()
    except:
        h, c = duration_model.init_hidden(batch_size=1)
        xl = len(x)
        x = x.view(1, -1, x.size(-1))
        duration_predicted = duration_model(x, [xl], h, c).data.numpy()
        duration_predicted = duration_predicted.reshape(-1, duration_predicted.shape[-1])

    # Apply denormalization
    duration_predicted = duration_predicted * Y_scale[ty] + Y_mean[ty]
    duration_predicted = np.round(duration_predicted)

    # Set minimum state duration to 1
    duration_predicted[duration_predicted <= 0] = 1
    hts_labels.set_durations(duration_predicted)

    return hts_labels