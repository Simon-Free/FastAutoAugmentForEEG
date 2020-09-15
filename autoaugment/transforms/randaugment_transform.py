import random

from ..construct_transforms import construct_transforms

# TODO : régler le problème du delay


def rand_transf(datum, params):
    n_transf = params["n_transf"]
    chosen_transf = []
    for i in range(n_transf):
        chosen_transf.append(random.choice(params["transform_list"]))
    chosen_transf.append("identity")
    constructed_chosen_transf = construct_transforms([chosen_transf], params)
    for transf in constructed_chosen_transf:
        datum = transf.transform(datum)
    return(datum)
