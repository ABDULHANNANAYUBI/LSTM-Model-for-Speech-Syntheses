class BinaryFileSource(FileDataSource):
    def __init__(self, data_root, dim, train):
        self.data_root = data_root
        self.dim = dim
        self.train = train
    def collect_files(self):
        files = sorted(glob(join(self.data_root, "*.bin")))
        files = files[:len(files)-5] # last 5 is real testset
        train_files, test_files = train_test_split(files, test_size=test_size,
                                                   random_state=random_state)
        if self.train:
            return train_files
        else:
            return test_files
    def collect_features(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, self.dim)



X = {"duration":{}, "acoustic": {}}
Y = {"duration":{}, "acoustic": {}}
utt_lengths = {"duration":{}, "acoustic": {}}
for ty in ["duration", "acoustic"]:
    for phase in ["train", "test"]:
        train = phase == "train"
        print(train)
        x_dim = duration_linguistic_dim if ty == "duration" else acoustic_linguisic_dim
        y_dim = duration_dim if ty == "duration" else acoustic_dim
        X[ty][phase] = FileSourceDataset(BinaryFileSource(join(DATA_ROOT, "X_{}".format(ty)),
                                                           dim = x_dim, train = "train"))
        Y[ty][phase] = FileSourceDataset(BinaryFileSource(join(DATA_ROOT, "Y_{}".format(ty)),
                                                        dim=y_dim,
                                                        train=  train))
        utt_lengths[ty][phase] = np.array([len(x) for x in X[ty][phase]], dtype=np.int)