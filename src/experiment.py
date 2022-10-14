import random
import numpy as np
import torch
import data
import models
import roc
import constants as const
import copy
import argparse
import notify

# default: --no-tune --wordvectors none --model cnn --subjpred
parser = argparse.ArgumentParser(description="Run Chinese Dyslexia Experiments")
parser.add_argument("--model", dest="model")
parser.add_argument("--roc", dest="roc", action="store_true")
parser.add_argument("--no-roc", dest="roc", action="store_false")
parser.add_argument("--tunesets", type=int, default=10)
parser.add_argument("--tune", dest="tune", action="store_true")
parser.add_argument("--no-tune", dest="tune", action="store_false")
parser.add_argument("--wordvectors", type=str, default="none")
parser.add_argument("--pretrain", dest="pretrain", action="store_true")
parser.add_argument("--subjpred", dest="batch_subjects", action="store_true")
parser.add_argument("--textpred", dest="batch_subjects", action="store_false")
parser.add_argument("--save-errors", dest="save_errors", type=argparse.FileType("w"))
parser.add_argument("--seed", dest="seed", type=int, default=42)
parser.add_argument("--cuda", dest="cudaid", default=0)
parser.set_defaults(tune=True)
parser.set_defaults(roc=True)
parser.set_defaults(batch_subjects=True)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.model == "cnn":
    MODEL_CLASS = models.CNNClassifier
elif args.model == "lstm":
    MODEL_CLASS = models.LSTMClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    device = torch.device(f'cuda:{args.cudaid}')


def getmeansd(dataset, batch: bool = False):
    if batch:
        tensors = [X for X, _, _ in dataset]
        tensors = torch.cat(tensors, axis=0)
        # remove padded tensors
        tensors = tensors[tensors.sum(dim=(1, 2)) != 0]
        means = torch.mean(tensors, dim=(0, 1))
        sd = torch.std(tensors, dim=(0, 1))
        return means, sd
    else:
        tensors = [X for X, _, _ in dataset]
        tensors = torch.cat(tensors, axis=0)
        # remove padded tensors
        tensors = tensors[tensors.sum(dim=1) != 0]
        means = torch.mean(tensors, 0)
        sd = torch.std(tensors, 0)
        return means, sd


def get_params(paramdict) -> dict:
    selected_pars = dict()
    for k in paramdict:
        selected_pars[k] = random.sample(list(paramdict[k]), 1)[0]
    return selected_pars


def _get_pretrained_models():
    for i in range(NUM_TUNE_SETS):
        pretrained_model = copy.deepcopy(MODEL_CLASS)
        yield pretrained_model.pretrain_model(
                    pretrain_dataset,
                    epochs=1,
                    device=device,
                    config=parameter_sample[i],
                )


def replicate_pretrained_models():
    for i in range(NUM_FOLDS**2):
        yield _get_pretrained_models()


NUM_FOLDS = 10
NUM_TUNE_SETS = args.tunesets
BATCH_SUBJECTS = args.batch_subjects
tune = args.tune

if args.wordvectors == "none":
    vectors = data.NoWordVectors
elif args.wordvectors == "bert":
    vectors = data.BertWordVectors
elif args.wordvectors == "bertmeandiff":
    vectors = data.BertMeandiffWordVectors
elif args.wordvectors == "bertpca":
    vectors = data.BertPCAWordVectors
elif args.wordvectors == "lingfeat":
    vectors = data.LingFeatWordVectors
else:
    raise ValueError(f"Unknown word vector '{args.wordvectors}'")

print("using word vectors: ", args.wordvectors)

# Prepare ROC Curves
if args.roc:
    Roc = roc.ROC(args.model, args.wordvectors, args.tune)

if args.save_errors is not None:
    args.save_errors.write("subj,y_pred,y_true\n")

if tune:
    used_test_params = []
    parameter_sample = [
        get_params(const.hyperparameter_space[args.model]) for _ in range(NUM_TUNE_SETS)
    ]

if args.pretrain:
    # load and preprocess data for pretraining
    # hack to reduce word vectors to same size without training mean difference encoder
    if args.wordvectors == "bertmeandiff":
        pretrain_vectors = data.BertPCAWordVectors
    else:
        pretrain_vectors = vectors
    pretraining_preprocessor = data.PretrainingDataPreprocessor(
        const.features,
        preprocessing={"gaze": lambda x: np.log(x) if x >= 1 else 0,
                       "osacdur": lambda x: np.log(x) if x >= 1 else 0,
                       "osacdx": lambda x: np.log(x) if x >= 1 else 0,
                       "osacl": lambda x: np.log(x) if x >= 1 else 0,
                       "isacdur": lambda x: np.log(x) if x >= 1 else 0,
                       "isacdx": lambda x: np.log(x) if x >= 1 else 0},
        word_vector_type=pretrain_vectors
    )
    pretraining_word_vector_model = pretraining_preprocessor.train_word_vector_model(folds=range(NUM_FOLDS))
    pretrain_dataset = data.PretrainingDataset(
        pretraining_preprocessor,
        pretraining_word_vector_model,
    )
    mean, sd = getmeansd(pretrain_dataset, batch=False)
    pretrain_dataset.standardize(mean, sd)
    if tune:
        print(f"Pretraining {NUM_TUNE_SETS} models...")
        pretrained_model_generator = replicate_pretrained_models()  # get generator of N_TUNE_SET models
        # pretrained_model_list = tee(pretrained_model_generator, NUM_FOLDS*NUM_FOLDS-1)  # replicate generator
    else:
        print('Pretraining one model')
        pretrained_model = copy.deepcopy(MODEL_CLASS)
        best_pretrained_model = pretrained_model.pretrain_model(
            pretrain_dataset,
            epochs=150,
            device=device,
            config=const.default_params[args.model],
        )


def main():
    try:
        # load and preprocess data for training
        preprocessor = data.EyetrackingDataPreprocessor(
            const.features,
            preprocessing={"gaze": lambda x: np.log(x) if x >= 1 else 0},
            word_vector_type=vectors,
            # word_vector_kwargs={
            #     "encoding_size": 20,
            #     "num_epochs": 30,
            #     "batch_size": 16,
            # },
            num_folds=NUM_FOLDS,
        )
        test_accuracies = []
        for test_fold in range(NUM_FOLDS):
            print("test fold ", test_fold)
            parameter_evaluations = np.zeros(shape=(NUM_FOLDS, NUM_TUNE_SETS))
            if tune:
                # Normal training / fine-tuning
                for dev_fold in range(NUM_FOLDS):
                    if args.pretrain:
                        pretrained_models = next(pretrained_model_generator)
                    if dev_fold == test_fold:
                        continue
                    train_folds = [
                        fold
                        for fold in range(NUM_FOLDS)
                        if fold != test_fold and fold != dev_fold
                    ]
                    word_vector_model = preprocessor.train_word_vector_model(
                            folds=train_folds
                        )
                    # When fine-tuning, we use the pre-trained word vector model (?)
                    train_dataset = data.EyetrackingDataset(
                        preprocessor,
                        word_vector_model,
                        folds=train_folds,
                        batch_subjects=BATCH_SUBJECTS,
                    )
                    mean, sd = getmeansd(train_dataset, batch=BATCH_SUBJECTS)
                    train_dataset.standardize(mean, sd)
                    dev_dataset = data.EyetrackingDataset(
                        preprocessor,
                        word_vector_model,
                        folds=[dev_fold],
                        batch_subjects=BATCH_SUBJECTS,
                    )
                    dev_dataset.standardize(mean, sd)
                    for tune_set in range(NUM_TUNE_SETS):
                        running_model = copy.deepcopy(MODEL_CLASS)
                        if tune_set%20 == 0:
                            print(f'tune set {tune_set}')
                        if args.pretrain:
                            pretrained_model = next(pretrained_models)
                        else:
                            pretrained_model = None
                        model = running_model.train_model(
                            train_dataset,
                            min_epochs=15,
                            max_epochs=200,
                            dev_data=dev_dataset,
                            pretrained_model=pretrained_model,
                            device=device,
                            config=parameter_sample[tune_set],
                        )
                        tune_accuracy = model.evaluate(
                            data=dev_dataset,
                            device=device,
                            metric="f1",
                            per_subj=BATCH_SUBJECTS,
                        )
                        parameter_evaluations[dev_fold, tune_set] = tune_accuracy
                # Select best parameter set
                mean_dev_accuracies = np.mean(parameter_evaluations, axis=0)
                best_parameter_set = np.argmax(mean_dev_accuracies)
                params_test = parameter_sample[best_parameter_set]
                # print(f'best performing parameter for fold ', test_fold, ": ", params_test)
                used_test_params.append(params_test)
                if args.pretrain:
                    pretrained_model = copy.deepcopy(MODEL_CLASS)
                    best_pretrained_model = pretrained_model.pretrain_model(
                                pretrain_dataset,
                                epochs=100,
                                device=device,
                                config=params_test,
                            )
                else:
                    best_pretrained_model = None
            else:  # (not tuning)
                params_test = const.default_params[args.model]
                best_pretrained_model = None
            # If tune: train using best feature set over dev sets, else: train using default parameters
            # Use fold next to test fold for early stopping
            running_model = copy.deepcopy(MODEL_CLASS)
            dev_fold = (test_fold + 1) % NUM_FOLDS
            train_folds = [
                fold for fold in range(NUM_FOLDS) if fold != test_fold and fold != dev_fold
            ]
            word_vector_model = preprocessor.train_word_vector_model(folds=train_folds)
            train_dataset = data.EyetrackingDataset(
                preprocessor,
                word_vector_model,
                folds=train_folds,
                batch_subjects=BATCH_SUBJECTS,
            )
            mean, sd = getmeansd(train_dataset, batch=BATCH_SUBJECTS)
            train_dataset.standardize(mean, sd)
            dev_dataset = data.EyetrackingDataset(
                preprocessor,
                word_vector_model,
                folds=[dev_fold],
                batch_subjects=BATCH_SUBJECTS
            )
            dev_dataset.standardize(mean, sd)
            test_dataset = data.EyetrackingDataset(
                preprocessor,
                word_vector_model,
                folds=[test_fold],
                batch_subjects=BATCH_SUBJECTS,
            )
            test_dataset.standardize(mean, sd)
            model = running_model.train_model(
                train_dataset,
                min_epochs=15,
                max_epochs=200,
                dev_data=dev_dataset,
                pretrained_model=best_pretrained_model,
                device=device,
                config=params_test,
            )
            print(f'test accuraccy fold ', test_fold)
            test_accuracy = model.evaluate(
                test_dataset,
                device=device,
                metric="all",
                print_report=True,
                per_subj=BATCH_SUBJECTS,
                save_errors=args.save_errors,
            )
            # print("test acc fold ", test_fold, " : ", test_accuracy)
            test_accuracies.append(test_accuracy)
            if args.roc:
                y_preds, y_trues = model.predict_probs(
                    test_dataset,
                    device=device,
                    per_subj=BATCH_SUBJECTS,
                )
                Roc.get_tprs_aucs(y_trues, y_preds, test_fold)

        if tune:
            print("used test params: ", used_test_params)
        print(
            "mean:",
            np.mean(test_accuracies, axis=0),
            "std:",
            np.std(test_accuracies, axis=0) / np.sqrt(NUM_FOLDS),
        )

        if args.roc:
            Roc.plot()
            Roc.save()
            print("auc: ", Roc.mean_auc, "+-", Roc.std_auc)
        pred_level = "subjectpred" if BATCH_SUBJECTS else "textpred"
        final_scores_mean = np.mean(test_accuracies, axis=0)
        final_scores_std = np.std(test_accuracies, axis=0) / np.sqrt(NUM_FOLDS)
        final_scores_mean = np.insert(final_scores_mean, 0, Roc.mean_auc, axis=0)
        final_scores_std = np.insert(final_scores_std, 0, Roc.std_auc, axis=0)
        out_str = ""
        with open(f"{args.model}_scores_{pred_level}_{args.wordvectors}.txt", "w") as f:
            for i in range(len(final_scores_mean)):
                out_str += f"${round(final_scores_mean[i],2):1.2f}\db{{{round(final_scores_std[i],2):1.2f}}}$"
                if i < len(final_scores_mean) - 1:
                    out_str += " & "
                else:
                    out_str += " \\\\ "
            f.write(out_str)

        notification = notify.Notification()
        notification.define_message(f'{args.model} {pred_level} {args.wordvectors}')
        notification.send_mail()
        return 0
    except Exception as e:
        print(e)
        notification = notify.Notification()
        notification.define_message(f'{e} for {args.model} subjpred:{BATCH_SUBJECTS} {args.wordvectors}', failed=True)
        notification.send_mail()


if __name__ == '__main__':
    raise SystemExit(main())


