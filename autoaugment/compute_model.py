
from .models.model_main_funcs import initialize_model, get_score, fit_model
from .retrieve_data import get_sample


def compute_experimental_result(model_args,
                                dataset_args,
                                train_dataset,
                                test_dataset,
                                sample_size):

    train_dataset.change_transform_list(dataset_args["transform_list"])
    score_list = []
    for i in range(model_args["n_cross_val"]):
        train_subset = get_sample(train_dataset,
                                  sample_size,
                                  random_state=i)
        model = initialize_model(model_args, train_subset)
        model = fit_model(model, model_args, train_subset)
        score_list.append(get_score(model, model_args, test_dataset))
    return score_list


# def compute_experimental_result_FastAutoAugment():
#     ops = augment_list(False)
#     space = {}
#     for i in range(args.num_policy):
#         for j in range(args.num_op):
#             space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
#             space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' % (i, j), 0.0, 1.0)
#             space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' % (i, j), 0.0, 1.0)

#     final_policy_set = []
#     total_computation = 0
#     reward_attr = 'top1_valid'      # top1_valid or minus_loss
#     for _ in range(1):  # run multiple times.
#         for cv_fold in range(cv_num):
#             name = "search_%s_%s_fold%d_ratio%.1f" % (C.get()['dataset'], C.get()['model']['type'], cv_fold, args.cv_ratio)
#             print(name)
#             register_trainable(name, lambda augs, rpt: eval_tta(copy.deepcopy(copied_c), augs, rpt))
#             algo = HyperOptSearch(space, max_concurrent=4*20, reward_attr=reward_attr)

#             exp_config = {
#                 name: {
#                     'run': name,
#                     'num_samples': 4 if args.smoke_test else args.num_search,
#                     'resources_per_trial': {'gpu': 1},
#                     'stop': {'training_iteration': args.num_policy},
#                     'config': {
#                         'dataroot': args.dataroot, 'save_path': paths[cv_fold],
#                         'cv_ratio_test': args.cv_ratio, 'cv_fold': cv_fold,
#                         'num_op': args.num_op, 'num_policy': args.num_policy
#                     },
#                 }
#             }

#             results = run_experiments(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True, resume=args.resume, raise_on_failed_trial=False)
#             print()
#             results = [x for x in results if x.last_result is not None]
#             results = sorted(results, key=lambda x: x.last_result[reward_attr], reverse=True)

