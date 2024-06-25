def search_vis_answers(result_dict, task_type, prompt_type, sampling_type):
    """
    Function purpose:
        - This function is implemented as the aggregation strategy, i.e., used to get the final answer and confidence based on the top-k results for every question, and aggregate all ensembles to get their corresponding scores: 
        - Prompt Strategy = "Vanilla" or "COT" or "Self-Probing"
        - or Prompt Strategy = "Multi-Step"
        
    Aggregation Strategy:
        - "AVG-Conf"
        - "Consistency"
        - "Pair-Rank" (implemented in another script)
        
    Hyperparameters:
        - result_dict: dict of all intermediate results
    """

    aggregation_strategy = ['avg_conf', 'avg_multistep_conf', "consistency"]
    score_dicts = {
      "real_answer": [],
      "scores": {}
    }
    
    for key in aggregation_strategy:
        score_dicts['scores'][key] = {"answer": [], "score": []}

    # for every question in the dataset, get their answers and their corresponding confidences -> can be multiple if num_ensemble > 1
    for key, value in result_dict.items():
        """ Example below:
        - key: question text
        - value: dict of all intermediate results
            - value['hint'] (keys: 'hint_response', 'generated_result', 'real_answer')
                - value['hint']['real_answer'] = {'options': 'A', 'option_number': 6}
                - value['hint']['generated_resutl'] 
                    - (keys: 'step1', 'step2', 'step3', 'final_answer', 'final_confidence')
                - value['hint']['generated_resutl']['step1'] = {'analysis': 'xxxx Confidence: 90;', 'confidence': 90}

        """
        real_answer = value['real_answer']
        if sampling_type == "misleading":
            predicted_answer = value['predicted_answer']
            predicted_conf = value['predicted_conf']
        
        # get predicted answers and confidences over multiple queries -> for ensemble
        # hint_answers = {"trai_0":{"0":"A", "1":"B"}, "trail_1":{}}
        # hint_confs = {"trai_0":{"0":90, "1":80}, "trail_1":{}}
        hint_answers = value['hint_answers']
        hint_confs = value['hint_confs']
        hint_multi_step_confs = value['hint_multi_step_confs']
        assert len(hint_answers) == len(hint_confs), "(len(hint_answers) should be equivalent to len(hint_confidences))"

        # process into a map: answer -> [conf1, conf2, conf3, ...]
        answer_confs_alltrails = {}
        for trail, ans in hint_answers.items():
            # sanity check ans is formatted correctly
            if ans is None:
                continue
            elif task_type == "multi_choice_qa":
                if ans not in normal_option_list:
                    continue
            if ans not in answer_confs_alltrails:
                answer_confs_alltrails[ans] = []
            # fill the answer-conflist map
            conf = hint_confs[trail] # get the corresponding confidence list for this 'trail_i' or 'hint_i'
            answer_confs_alltrails[ans].append(conf)
        
        answer_stepconfs_for_alltrails = {}
        hint_step_confs = {}
        if prompt_type == "multi_step":
            for trail, step_confs in hint_multi_step_confs.items():
                confidence_product = 1
                for step_idx, step_result in step_confs.items():
                    step_confidence = step_result['confidence']
                    confidence_product *= step_confidence
                ans = hint_answers[trail]
                if ans not in answer_stepconfs_for_alltrails:
                    answer_stepconfs_for_alltrails[ans] = []
                answer_stepconfs_for_alltrails[ans].append(confidence_product)
                hint_step_confs[trail] = confidence_product
            
        ################### AVG-CONF ####################
        # compute consistency
        def compute_consistency_score(hint_answers, sampling_type):
            """every query has a answer, find the most frequent answer and its frequency -> consistency score"""
            top_1_ans = [answer for _, answer in hint_answers.items()]
            counter = Counter(top_1_ans)
            total = len(top_1_ans)
            # compute the frequency of each answer
            frequencies = {key: value / total for key, value in counter.items()}
            # find the most frequent answer and its frequency
            if sampling_type == "misleading":
                most_freq_ans = predicted_answer
                most_freq_score = frequencies[most_freq_ans]
                return most_freq_ans, most_freq_score
            
            most_freq_ans = max(frequencies, key=frequencies.get)
            most_freq_score = frequencies[most_freq_ans]
            return most_freq_ans, most_freq_score
        
        consistency_answer, consistency_score = compute_consistency_score(hint_answers, sampling_type)
        score_dicts['scores']['consistency']['answer'].append(consistency_answer)
        score_dicts['scores']['consistency']['score'].append(consistency_score)

        
        # compute average confidence for every possible answer
        def compute_avg_confidence(hint_confs, answer_confs_alltrails, sampling_type):
            conf_list = [conf for conf in hint_confs.values()]
            conf_sum = np.sum(conf_list)
            average_confs = {ans: sum(conf_lists)/conf_sum for ans, conf_lists in answer_confs_alltrails.items()}
            
            if sampling_type == "misleading":
                avg_conf_option = predicted_answer
                avg_confidence = average_confs[avg_conf_option]
                return avg_conf_option, avg_confidence
            
            avg_conf_option = max(average_confs, key=average_confs.get)
            avg_confidence = average_confs[avg_conf_option]
            return avg_conf_option, avg_confidence
        
        avg_conf_option, avg_confidence = compute_avg_confidence(hint_confs, answer_confs_alltrails, sampling_type)
        
        if prompt_type == "multi_step":
            avg_step_conf_option, avg_step_confidence = compute_avg_confidence(hint_step_confs, answer_stepconfs_for_alltrails, sampling_type)
        
        if task_type == "open_number_qa":
            real_answer = float(real_answer)
            consistency_answer = float(consistency_answer)
            avg_conf_option = float(avg_conf_option)
            if prompt_type == "multi_step":
                avg_step_conf_option = float(avg_step_conf_option)
            
        elif task_type == 'multi_choice_qa':
            if isinstance(real_answer, int):
                real_answer = option_list[real_answer]    

            
        score_dicts["real_answer"].append(real_answer)
        score_dicts['scores']['avg_conf']['answer'].append(avg_conf_option)
        score_dicts['scores']['avg_conf']['score'].append(avg_confidence)     
        if prompt_type == "multi_step":
            score_dicts['scores']['avg_multistep_conf']['answer'].append(avg_step_conf_option)
            score_dicts['scores']['avg_multistep_conf']['score'].append(avg_step_confidence)      

        
    print("Total count: ", len(score_dicts['real_answer']))  
    return score_dicts


        
score_dict = search_vis_answers(data, args.task_type, prompt_type=args.prompt_type, sampling_type=args.sampling_type)   

import logging
import torch
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from netcal.metrics import ECE

import numpy as np

pruna_logger = logging.getLogger("quant_logger")

def compute_conf_metrics(y_true, y_confs):
    result_matrics = {}
    # ACC
    accuracy = sum(y_true) / len(y_true)
    print("accuracy: ", accuracy)
    result_matrics['acc'] = accuracy

    # use np to test if y_confs are all in [0, 1]
    assert all([x >= 0 and x <= 1 for x in y_confs]), y_confs
    y_confs, y_true = np.array(y_confs), np.array(y_true)

    # AUCROC
    roc_auc = roc_auc_score(y_true, y_confs)
    print("ROC AUC score:", roc_auc)
    result_matrics['auroc'] = roc_auc

    # AUPRC-Positive
    auprc = average_precision_score(y_true, y_confs)
    print("AUC PRC Positive score:", auprc)
    result_matrics['auprc_p'] = auprc

    # AUPRC-Negative
    auprc = average_precision_score(1- y_true, 1 - y_confs)
    print("AUC PRC Negative score:", auprc)
    result_matrics['auprc_n'] = auprc

    # AURC from https://github.com/IML-DKFZ/fd-shifts/tree/main
    # aurc = area_under_risk_coverage_score(y_confs, y_true)
    # result_matrics['aurc'] = aurc
    # print("AURC score:", aurc)

    # ECE
    n_bins = 10
    # diagram = ReliabilityDiagram(n_bins)
    ece = ECE(n_bins)
    ece_score = ece.measure(np.array(y_confs), np.array(y_true))
    print("ECE:", ece_score)
    result_matrics['ece'] = ece_score

    return result_matrics