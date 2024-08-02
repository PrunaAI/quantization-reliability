QUANTO = {
    "name": "QUANTO",
    "quantize_method": "QUANTO",
    "num_bits": 8,
    "weights": "qint8",
    "activations": "qfloat8",
}

QUANTO_CALIB = {
    "name": "QUANTO-CALIB",
    "quantize_method": "QUANTO",
    "num_bits": 8,
    "weights": "qint8",
    "activations": "qfloat8",
    "n_samples": 128,
    "momentum": 0.9,
}

QUANTO_QAT = {
    "name": "QUANTO-QAT",
    "quantize_method": "QUANTO",
    "num_bits": 8,
    "weights": "qint8",
    "activations": "qfloat8",
    "train_samples": 128,
    "lr": 1e-4,
}
