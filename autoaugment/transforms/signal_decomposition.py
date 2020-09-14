import random


def merge_two_signals(datum, params):

    signal = datum.X
    y = datum.y
    train_sample = params["train_sample"]
    label_index_dict = params["label_index_dict"]
    other_signal_index = random.choice(label_index_dict[y])
    other_signal = train_sample[other_signal_index][1]
    final_signal = (1 - params["magnitude"]) * \
        signal + params["magnitude"] * other_signal
    datum.X = final_signal

    return datum
