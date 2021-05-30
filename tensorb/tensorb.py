# conda install -c conda-forge tensorboard
# conda install -c conda-forge protobuf

# pip install tensorboardX

# tensorboard --logdir=runs

from tensorboardX import SummaryWriter
import os, csv, sys
from Utils import commons_utils
import numpy as np
import ast, json

# Ignore experiments seeds that did not converge
IGNORE_FAILED_SEEDS = True


def read_experiment_stats(mapping, key, filepath, triple_stat=False):
    if not os.path.exists(filepath): return mapping
    with open(filepath, 'r') as f:
        values = [[float(ast.literal_eval(v)[0] if triple_stat else v) for v in x] for x in csv.reader(f)]
        if key not in mapping: mapping[key] = []
        mapping[key].append(values)
    return mapping


def main():
    file_to_section_map = {
        'd_accuracies.csv': 'gan/dis_accuracy',
        'times.csv': 'gan/exec_time',
        'd_losses.csv': 'gan/d_loss',
        'g_losses.csv': 'gan/g_loss',
        'lls_avg.csv': 'loglikelihood/nll_avg',
        'lls_avg_conditional.csv': 'loglikelihood/conditional_nll_avg',
        'pre_classifier_accuracies.csv': 'classifier_metric/classifier_accuracy_avg',
        'fid_scores.csv': 'gan/fid_score'
    }

    for experiment in os.listdir('.'):
        if 'experiment_' not in experiment or 'cifar10' in experiment: continue
        if len(sys.argv) > 1 and sys.argv[1] not in experiment: continue
        if not os.path.isdir(experiment): continue

        print('experiment', experiment)
        output_pre_trained_classifier_accuracies = []
        if os.path.exists(os.path.join(experiment, 'output')):
            output_pre_trained_classifier_accuracies = np.mean(commons_utils.load_metric_evaluation(os.path.join(experiment, 'output')), axis=0)

        for model in os.listdir(experiment):
            if not os.path.isdir(os.path.join(experiment, model)): continue
            print('model', model)
            # Writer will output to ./runs/ directory by default
            writer = SummaryWriter(logdir=os.path.join('runs', experiment+'_'+model))

            metrics_map = {}

            converged = 0
            float_tensor_worked = 0
            
            for seed in os.listdir(os.path.join(experiment, model)):
                if not os.path.isdir(os.path.join(experiment, model, seed)): continue
                print('seed', seed)

                ################################
                ### Converged?
                convergence_worked = True
                
                if "mnist_condgan" in experiment:
                    classifier_accuracies_file = os.path.join(experiment,model,seed, commons_utils.DIR_STATS,'pre_classifier_accuracies.csv')
                    if os.path.exists(classifier_accuracies_file):
                        convergence_worked = commons_utils.model_converged(classifier_accuracies_file, lambda x: float(x[0][-1]) > 90)
                        if convergence_worked: converged += 1
                elif "celeba-dcgan" in experiment:
                    fid_scores_file = os.path.join(experiment,model,seed, commons_utils.DIR_STATS,'fid_scores.csv')
                    if os.path.exists(fid_scores_file):
                        convergence_worked = commons_utils.model_converged(fid_scores_file, lambda x: float(x[0][-1]) < 30)
                        if convergence_worked: converged += 1
                else: raise Exception("missing convergence threshold setup for this experiment")
                ### Worked with float tensors?
                config_file = os.path.join(experiment, model, seed, 'config.json')
                if os.path.exists(config_file):
                    with open(config_file) as f:
                        try:
                            if json.load(f)["DOUBLE_TENSORS"] == 0: float_tensor_worked += 1
                        except:
                            continue
                
                ### next seed?
                if IGNORE_FAILED_SEEDS and not convergence_worked: continue
                ################################
                

                for stat_name in os.listdir(os.path.join(experiment, model, seed, commons_utils.DIR_STATS)):
                    # ignore the following stat files
                    if(stat_name in ['x.csv', 'lls_std.csv', 'lls_std_conditional.csv']): continue

                    if any(aux in stat_name for aux in ['Gen_', 'Dis_', 'dis_', 'gen_']):    
                        tensorb_name_with_section = 'parameters/' + stat_name.split('.')[0]
                        triple_stat = True
                    else:
                        tensorb_name_with_section = file_to_section_map[stat_name]
                        triple_stat = False
                    
                    file_path = os.path.join(experiment, model, seed, commons_utils.DIR_STATS, stat_name)
                    metrics_map = read_experiment_stats(metrics_map, tensorb_name_with_section, file_path, triple_stat=triple_stat)

            if 'classifier_metric/classifier_accuracy_avg' not in metrics_map:
                metrics_map = read_experiment_stats(metrics_map, 'classifier_metric/classifier_accuracy_avg', os.path.join(experiment, model, 'classifier_acc_avgs.csv'))
                metrics_map = read_experiment_stats(metrics_map, 'classifier_metric/classifier_accuracy_std', os.path.join(experiment, model, 'classifier_acc_stds.csv'))
            else:
                metrics_map['classifier_metric/classifier_accuracy_std'] = np.array([np.std(metrics_map['classifier_metric/classifier_accuracy_avg'], axis=0)])

            if 'loglikelihood/nll_avg' in metrics_map and len(metrics_map['loglikelihood/nll_avg']) > 0:
                metrics_map['loglikelihood/nll_std'] = np.std(metrics_map['loglikelihood/nll_avg'], axis=0)[0]
            if 'loglikelihood/conditional_nll_avg' in metrics_map and len(metrics_map['loglikelihood/conditional_nll_avg']) > 0:
                metrics_map['loglikelihood/conditional_nll_std'] = np.std(metrics_map['loglikelihood/conditional_nll_avg'], axis=0)[0]

            for key in metrics_map.keys():
                if len(metrics_map[key]) == 0: continue

                # activation parameters statistics
                if 'parameters' in key:
                    for step in range(len(metrics_map[key][0][0])):
                        writer.add_histogram('hist_'+key, np.array(metrics_map[key])[:,0,step], step)
                
                # the mean of all seeds
                if key not in ['loglikelihood/nll_std', 'loglikelihood/conditional_nll_std']:
                    metrics_map[key] = np.mean(metrics_map[key], axis=0)[0] # [0] because it returns [[n1,n2,...,n]]
                
                if key == 'classifier_metric/classifier_accuracy_avg' and len(output_pre_trained_classifier_accuracies) > 0:
                    for n_iter in range(len(metrics_map[key])):
                        writer.add_scalar('classifier_metric/diff_to_output', metrics_map[key][n_iter]-output_pre_trained_classifier_accuracies[n_iter], n_iter)
                
                for n_iter in range(len(metrics_map[key])):
                    writer.add_scalar(key, metrics_map[key][n_iter], n_iter)
            
            writer.add_scalar("experiment/convergence", converged, 0)
            writer.add_scalar("experiment/used_float_tensors", float_tensor_worked, 0)

            writer.close()

if __name__ == '__main__':
    main()