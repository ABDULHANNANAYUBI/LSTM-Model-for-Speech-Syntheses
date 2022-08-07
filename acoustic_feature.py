from ast import AnnAssign
import numpy as np
def SetDuration():
    duration_linguistic_dim = np.zeros(10, dtype=np.float64)
    for ty in ["duration", "acoustic"]:
        for phase in ["train", "test"]:
            train = phase == "train"
            x_dim = duration_linguistic_dim if ty == "duration" else acoustic_linguisic_dim
            y_dim = duration_dim if ty == "duration" else acoustic_dim
            X[ty][phase] = PaddedFileSourceDataset(BinaryFileSource(join(DATA_ROOT, "X_{}".format(ty)), #we should use diff. libraries. 
                                                        dim=x_dim,
                                                        train=train),
                                                np.max(utt_lengths[ty][phase]))
            Y[ty][phase] = PaddedFileSourceDataset(BinaryFileSource(join(DATA_ROOT, "Y_{}".format(ty)),
                                                        dim=y_dim,
                                                        train=train),
                                                np.max(utt_lengths[ty][phase]))