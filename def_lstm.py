def train_lstm(model, optimizer, X, Y, X_min, X_max, Y_mean, Y_scale,
          utt_lengths, cache_size=1000):
    if use_cuda:
        model = model.cuda()

    X_train, X_test = X["train"], X["test"]
    Y_train, Y_test = Y["train"], Y["test"]
    train_lengths, test_lengths = utt_lengths["train"], utt_lengths["test"]

    # Sequence-wise train loader
    X_train_cache_dataset = MemoryCacheDataset(X_train, cache_size) // here it should change
    Y_train_cache_dataset = MemoryCacheDataset(Y_train, cache_size) // here it should change 
    train_dataset = PyTorchDataset(X_train_cache_dataset, Y_train_cache_dataset, train_lengths,
                                  X_min, X_max, Y_mean, Y_scale)
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)

    # Sequence-wise test loader
    X_test_cache_dataset = MemoryCacheDataset(X_test, cache_size)
    Y_test_cache_dataset = MemoryCacheDataset(Y_test, cache_size)
    test_dataset = PyTorchDataset(X_test_cache_dataset, Y_test_cache_dataset, test_lengths,
                                 X_min, X_max, Y_mean, Y_scale)
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)

    dataset_loaders = {"train": train_loader, "test": test_loader}

    # Training loop
    criterion = nn.MSELoss()
    model.train()
    print("Start utterance-wise training...")
    loss_history = {"train": [], "test": []}
    for epoch in tnrange(nepoch):
        for phase in ["train", "test"]:
            running_loss = 0
            for x, y, lengths in dataset_loaders[phase]:
                # Sort by lengths . This is needed for pytorch's PackedSequence
                sorted_lengths, indices = torch.sort(lengths.view(-1), dim=0, descending=True)
                sorted_lengths = sorted_lengths.long().numpy()
                # Get sorted batch
                x, y = x[indices], y[indices]
                # Trim outputs with max length
                y = y[:, :sorted_lengths[0]]

                # Init states
                h, c = model.init_hidden(len(sorted_lengths))
                if use_cuda:
                    x, y = x.cuda(), y.cuda()
                if use_cuda:
                    h, c = h.cuda(), c.cuda()
                x, y = Variable(x), Variable(y)
                optimizer.zero_grad()

                # Do apply model for a whole sequence at once
                # no need to keep states
                y_hat = model(x, sorted_lengths, h, c)
                loss = criterion(y_hat, y)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
            loss_history[phase].append(running_loss / (len(dataset_loaders[phase])))

    return loss_history