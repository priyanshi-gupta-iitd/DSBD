import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pandas as pd
import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
from datasets import load_dataset

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from sampling import beam_speculative_sampling
from sampling import random_width_beam_sampling
from sampling.models.modeling_llama import LlamaForCausalLM
from sampling.models.modeling_opt import OPTForCausalLM
from sampling.utils import exact_match_references, execution_accuracy_references 
from sampling.utils import extract_first_function
from sampling.utils import execution_accuracy

from constraints import build_constraint_manager, compute_goodput_row
from constraints.z3_schema import SchemaFacts

import json
from time import process_time_ns
from tqdm import tqdm
import time
import numpy as np
import random
import subprocess
import pickle
import csv
import evaluate as hf_evaluate
#from pyJoules.energy_meter import measure_energy
hf_token = os.environ.get('HFTOKEN', None)

def find_fields_MYSQL_like(db_name, spider_schema):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "Table " +name+ ', columns = ['
    for index, row in group.iterrows():
      output += row[" Field Name"]+','
    output = output[:-1]
    output += "]\n"
  return output

def creatiing_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign


def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--approx_model_name', type=str, default="facebook/opt-125m")
    parser.add_argument('--target_model_name', type=str, default="facebook/opt-350m")
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=123, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--log_file', type=str, default="logs/log.txt")
    parser.add_argument('--dataset', type=str, default='wmt')
    parser.add_argument('--max_seconds', type=int, default=7200, help='timeout seconds')
    parser.add_argument('--top_k', type=int, default=10, help='k for top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.8, help='p for top-p sampling')
    parser.add_argument('--num_inputs', type=int, default=100, help='the number of inputs for each dataset')
    parser.add_argument('--constraints', type=str, default='none',
                        choices=['none', 'xgrammar', 'z3', 'both'],
                        help='decoding-time constraints')
    parser.add_argument('--constraint_ablation', action='store_true', default=False,
                        help='run 2x2 ablation: none/xgrammar/z3/both with fixed DSBD config')
    parser.add_argument('--fixed_dsbd', action='store_true', default=False,
                        help='use fixed DSBD hyperparams (width=4,gamma=3,w_thres=0.9) instead of full grid')
    parser.add_argument('--skip_baselines', action='store_true', default=False,
                        help='skip AR / vanilla SD / beam baselines')
    parser.add_argument('--metrics_csv', type=str, default='logs/constraint_metrics.csv',
                        help='per-example metrics CSV for goodput analysis')
    parser.add_argument('--dsbd_width', type=int, default=4)
    parser.add_argument('--dsbd_gamma', type=int, default=3)
    parser.add_argument('--dsbd_w_thres', type=float, default=0.9)
    parser.add_argument('--dsbd_min_w', type=int, default=1)
    parser.add_argument('--dsbd_extra_sample_cnt', type=int, default=1)
    args = parser.parse_args()
    return args


def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    

def get_score(output, target_model, input_len):
    with torch.no_grad():
        if target_model.config.is_encoder_decoder == False:
            logits = target_model(output).logits
            logits = logits[:,:-1,:]
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            logits = torch.gather(logits,
                          dim = -1,
                          index = output[:,1:,None])
            if logits.isnan().any():
                print(logits.size())
                print(logits)
                print(old_logits)
                xxx = input()

            return torch.mean(logits[:,input_len-1:,:])
        else:
            logits = target_model(output[:, :input_len], decoder_input_ids=output[:,input_len:]).logits
            logits = logits[:, :-1, :]
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            logits = torch.gather(logits,
                                  dim = -1,
                                  index = output[:, input_len+1:, None])
            return torch.mean(logits)

def get_total_power(outputs, t1, t2, fname):
    if fname is not None:
        with open(fname, 'wb') as f:
            pickle.dump((outputs, t1, t2), f)
    x = [out.strip().split() for out in outputs]
#    for xx in x:
#        if len(xx) < 2:
#            print(outputs)
#            print(x)
#            print(xx)
    x = [[float(xx[0]), float(xx[1])] for xx in x if len(xx) >= 2] # it seems possible that the last output of nvidia-smi is missing
    total_power = 0
    first_one = True
    for timestamp, power in x:
        if timestamp > t1 and timestamp < t2:
            if first_one:
                first_one = False
            else:
                total_power += power
    return total_power


def _METRICS_FIELDS():
    return [
        "method", "constraints", "example_idx", "db_id", "pred_sql",
        "committed_tokens", "tokens_proposed", "tokens_accepted",
        "tokens_constraint_rejected", "xgrammar_rejects", "z3_rejects",
        "wall_time_s", "xgrammar_time_s", "z3_time_s",
        "throughput", "goodput", "goodput_correct", "useful_frac",
        "n_useful", "n_useful_correct", "executable", "exec_correct", "exec_acc",
        "acc_len_mean", "acc_rate",
        "width", "gamma", "w_thres", "min_w", "extra_sample_cnt",
    ]


def _append_metrics_row(csv_path, row, fieldnames=None):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    if fieldnames is None:
        fieldnames = _METRICS_FIELDS()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _decode_pred(tokenizer, output, input_len, dataset_name):
    text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    if dataset_name == "squad":
        return text.split("\n")[0]
    if dataset_name == "spider":
        return text.split(";")[0]
    return text


def _schema_facts_for_example(dataset_name, output_dataset, idx, spider_schema):
    if dataset_name != "spider" or spider_schema is None:
        return None
    db_id = output_dataset[idx].split("[SQL]")[0]
    return SchemaFacts.from_spider_frames(db_id, spider_schema)


def _make_constraint_manager(mode, tokenizer, schema_facts):
    if mode == "none":
        return None
    return build_constraint_manager(mode, tokenizer=tokenizer, schema_facts=schema_facts)


#@measure_energy
def evaluate(approx_model_name, 
        target_model_name, 
        dataset_name, 
        num_tokens=20, 
        top_k = 10,
        top_p = 0.9,
        num_inputs = 100,
        max_seconds = 7200,
        random_seed = None, 
        verbose = False, 
        log_file = "logs/log.txt",
        constraints = "none",
        constraint_ablation = False,
        fixed_dsbd = False,
        skip_baselines = False,
        metrics_csv = "logs/constraint_metrics.csv",
        dsbd_width = 4,
        dsbd_gamma = 3,
        dsbd_w_thres = 0.9,
        dsbd_min_w = 1,
        dsbd_extra_sample_cnt = 1,
        ):
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    log_f = open(log_file, 'w')
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True, token=hf_token)

    tokenizer2 = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True, token=hf_token)
    print(approx_model_name, file=log_f)
    print(target_model_name, file=log_f)

    vocab1 = tokenizer.get_vocab()
    vocab2 = tokenizer2.get_vocab()
    if vocab1 == vocab2:
        print("Vocabularies are the same. Proceed")
    else:
        print("Vocabularies are different.")
        print("Vocabularies are different.", file=log_f)
        return
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    if 'Llama-3' in approx_model_name:
        small_model = LlamaForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.bfloat16,
                                                       device_map="auto",
                                                       offload_folder="offload",
                                                       trust_remote_code=True,
                                                       token = hf_token,
                                                       )
        tokenizer.pad_token_id = tokenizer.eos_token_id


    elif 'llama' in approx_model_name:
        small_model = LlamaForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True,
                                                       token=hf_token)
    elif 'opt' in approx_model_name:
        small_model = OPTForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True,
                                                       token=hf_token)
    else:
        raise NotImplementedError

    if 'Llama-3' in target_model_name:
         large_model = LlamaForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float32,
                                                       device_map="auto",
                                                       offload_folder="offload",
                                                       trust_remote_code=True,
                                                       token = hf_token,
                                                       )
        
    elif 'llama' in target_model_name:
        large_model = LlamaForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       offload_folder="offload",
                                                       trust_remote_code=True,
                                                       token=hf_token,
                                                       )
                                                       #token=hf_token)


    elif 'opt' in target_model_name:
        large_model = OPTForCausalLM .from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       offload_folder="offload",
                                                       trust_remote_code=True,
                                                       token=hf_token)
    else:
        raise NotImplementedError

    repeats = 1
    spider_schema = None
    
    if dataset_name == 'squad':
        dataset = load_dataset('squad', split='validation')
        examples = """[INST] <<SYS>> You need to answer the question using the exact words from the context. Below are some examples of how to answer questions based on context<</SYS>>
Example 1
Context: Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. 
Question: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?
Answer: Saint Bernadette Soubirous

Now, answer the following question[/INST]
"""
        input_texts = [examples + 
                       "Context: " + s["context"] + '\n'+
                       "Question: " + s["question"] + ' \n'+
                       "Answer:" for s in dataset]
        input_dataset = [tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True) for text in input_texts]
        output_dataset = [s["answers"]["text"] for s in dataset]
        iid_params = [(4,2)]
    elif dataset_name == 'spider':
        iid_params = [(4,2)]

        import json
        dataset = json.load(open("spider/spider/dev.json"))
        spider_schema,spider_primary,spider_foreign = creatiing_schema("./spider/spider/tables.json")

        examples = """[INST] <<SYS>> You are a SQL expert. You need to write the correct SQL based on the user question and database schemas. Below are some examples <</SYS>>
Example 
Schema:
Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Question: "How many heads of the departments are older than 56 ?"
SQL: SELECT count(*) FROM head WHERE age  >  56; 

"""
        input_texts = [examples + 
                       "Schema:\n" + find_fields_MYSQL_like(s["db_id"], spider_schema) + "\n" + 
                       "Question: " + s["question"] + "\n" + 
                       "SQL:" for s in dataset]
        input_dataset = [tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True) for text in input_texts]
        output_dataset = [s["db_id"] + "[SQL]" + s["query"] for s in dataset] 
    else:
        raise RuntimeError(f"Unrecognized dataset {dataset_name}. If you want to run MT-Bench, please use download repo of MT-Bench and modify the decoding algorithm to our our algorithm")


    length_interval = [100000]

    prefix = "./logs/"
    approx_model_name = os.path.basename(approx_model_name)
    target_model_name = os.path.basename(target_model_name)


    if dataset_name == 'squad':
        em = exact_match_references
    elif dataset_name == 'spider':
        em = execution_accuracy_references
    else:
        em = None

    # Constraint ablation / fixed DSBD: prefer focused experiment path
    use_fixed = fixed_dsbd or constraint_ablation or constraints != "none"
    if constraint_ablation:
        constraint_modes = ["none", "xgrammar", "z3", "both"]
    else:
        constraint_modes = [constraints]

    ori_output_dataset = output_dataset
    for i in range(repeats):
        u = 100000
        l = 0
        # This is to remove possible inputs whose length is too long
        ds = [pt for pt in input_dataset if (pt.size(-1) < u and pt.size(-1) >= l)]
        ds = ds[:num_inputs]
        output_dataset = ori_output_dataset[:num_inputs]

#        output_dataset = [ori_output_dataset[12] for k in range(100)]

        print(f'input length {l}-{u}, {len(ds)} data in total')
        total_input_tokens = sum([d.size(1) for d in ds])
        print('total_input_tokens', total_input_tokens)

        def log_both(msg):
            print(msg)
            print(msg, file=log_f)

        ######################################################################
        # Large-model AR baseline (optional)
        ######################################################################
        large_model_cnt = 0
        if not skip_baselines:
            total_time = 0
            total_token = 0
            scores = []
            pred_seq = []
            P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
            t1 = time.time()
            for ex_i, input_ids in enumerate(tqdm(ds)):
                large_model_cnt += 1
                input_ids = input_ids.to(torch_device)
                facts = _schema_facts_for_example(dataset_name, output_dataset, ex_i, spider_schema)
                # AR baseline uses first constraint mode only when not ablating all for AR;
                # keep unconstrained AR as quality ceiling unless constraints requested.
                cm = _make_constraint_manager(constraint_modes[0] if constraint_modes[0] != "none" else "none",
                                              tokenizer, facts)
                t = process_time_ns()
                output, details = autoregressive_sampling(
                    input_ids, large_model, num_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    top_k=top_k, top_p=top_p, pad_token_id=tokenizer.pad_token_id,
                    constraint_manager=cm, details=True)
                wall = process_time_ns() - t
                total_time += wall
                committed = len(output[0]) - input_ids.size(1)
                total_token += committed
                score = get_score(output, large_model, input_ids.size(1))
                scores.append(score.item())
                pred = _decode_pred(tokenizer, output, input_ids.size(1), dataset_name)
                pred_seq.append(pred)
                db_id = output_dataset[ex_i].split("[SQL]")[0] if dataset_name == "spider" else None
                exec_acc = None
                if dataset_name == "spider":
                    gt = output_dataset[ex_i].split("[SQL]")[1]
                    exec_acc = float(max(execution_accuracy(db_id, pred, gt), 0))
                row = compute_goodput_row(
                    method="ar_target",
                    constraints=constraint_modes[0] if cm is not None else "none",
                    example_idx=ex_i,
                    db_id=db_id,
                    pred_sql=pred if dataset_name == "spider" else "",
                    reference=output_dataset[ex_i] if dataset_name == "spider" else None,
                    committed_tokens=details.get("committed_tokens", committed),
                    tokens_proposed=details.get("tokens_proposed", committed),
                    tokens_accepted=details.get("tokens_accepted", committed),
                    tokens_constraint_rejected=details.get("tokens_constraint_rejected", 0),
                    xgrammar_rejects=details.get("xgrammar_rejects", 0),
                    z3_rejects=details.get("z3_rejects", 0),
                    wall_time_ns=details.get("wall_time_ns", wall),
                    xgrammar_time_ns=details.get("xgrammar_time_ns", 0),
                    z3_time_ns=details.get("z3_time_ns", 0),
                    exec_acc=exec_acc,
                )
                _append_metrics_row(metrics_csv, row)
                if total_time / 1e9 > max_seconds:
                    log_both(f'terminated at {large_model_cnt}')
                    break
            t2 = time.time()
            P.kill()
            P.wait()
            outputs = P.stdout.readlines()
            power_total = get_total_power(outputs, t1, t2, None)
            log_both(f'\nlarge model total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/max(total_token,1)} s/token, prob_score = {np.mean(scores)}')
            log_both(f'total power consumption: {power_total}')
            if em is not None and pred_seq:
                em_score = em(predictions=pred_seq, references=output_dataset[:large_model_cnt])
                log_both(f'em score = {em_score}')
        else:
            large_model_cnt = len(ds)

        ######################################################################
        # Vanilla speculative decoding (optional, unconstrained details)
        ######################################################################
        if not skip_baselines and not use_fixed:
            total_time = 0
            total_token = 0
            approx_time = 0
            target_time = 0
            other_time = 0
            total_acc_len = 0
            acc_rate = []
            target_times = 0
            approx_times = 0
            scores = []
            pred_seq = []
            cnt = 0
            P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
            t1 = time.time()
            target_model_time = 0
            target_pre_cache_time = 0
            target_post_prob_time = 0
            for input_ids in tqdm(ds):
                cnt += 1
                input_ids = input_ids.to(torch_device)
                t = process_time_ns()
                output, details = speculative_sampling(input_ids, small_model, large_model, 
                        eos_token_id = tokenizer.eos_token_id,
                        pad_token_id = tokenizer.pad_token_id,
                        max_len = num_tokens, 
                        top_k = top_k, top_p=top_p, random_seed = None, details=True)
                total_time += process_time_ns() - t
                total_token += len(output[0])- input_ids.size(1)
                approx_time += details['approx_time']
                target_time += details['target_time']
                other_time += details['other_time']
                total_acc_len += np.sum(details['acc_len'])
                acc_rate.append(details['acc_rate'])
                target_times += details['target_call_times']
                approx_times += details['approx_call_times']
                target_model_time += details['target_model_time']
                target_pre_cache_time += details['target_pre_cache_time']
                target_post_prob_time += details['target_post_prob_time']
                score = get_score(output, large_model, input_ids.size(1))
                scores.append(score.item())
                pred_seq.append(_decode_pred(tokenizer, output, input_ids.size(1), dataset_name))
                if total_time / 1e9 > max_seconds:
                    log_both(f'terminated at {cnt}')
                    break
            t2 = time.time()
            P.kill(); P.wait()
            outputs = P.stdout.readlines()
            fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_ss.pkl")
            power_total = get_total_power(outputs, t1, t2, fname)
            log_both(f'\nspeculative decoding total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/max(total_token,1)} s/token')
            log_both(f"average accepted len {total_acc_len/max(target_times,1)}, acc rate {np.mean(acc_rate) if acc_rate else 0}")
            if em is not None and pred_seq:
                em_score = em(predictions=pred_seq, references=output_dataset[:cnt])
                log_both(f'em score = {em_score}')

        ######################################################################
        # DSBD — fixed config under constraints, else optional full grid
        ######################################################################
        if use_fixed:
            width_list = [dsbd_width]
            extra_list = [dsbd_extra_sample_cnt]
            thres_list = [dsbd_w_thres]
            gamma_list = [dsbd_gamma]
            minw_list = [dsbd_min_w]
        else:
            width_list = [2, 3, 4, 5, 6]
            extra_list = [1, -1]
            thres_list = [0.7, 0.9]
            gamma_list = [2, 3]
            minw_list = [1, 2, 3]

        for cmode in constraint_modes:
          for width in width_list:
            for extra_sample_cnt in extra_list:
              if (not use_fixed) and extra_sample_cnt == 1 and width > 3:
                  continue
              for w_thres in thres_list:
               for gamma in gamma_list:
                for min_w in minw_list:
                  if min_w > width:
                        continue
                  num_beams = width
                  total_time = 0
                  total_token = 0
                  approx_time = 0
                  target_time = 0
                  other_time = 0
                  total_acc_len = 0
                  compute_expect_time = 0
                  acc_rate = []
                  target_times = 0
                  approx_times = 0
                  scores = []
                  pred_seq = []
                  cnt = 0
                  sum_proposed = 0
                  sum_useful = 0
                  sum_wall = 0
                  P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
                  t1 = time.time()
                  expect_cnt_list = []

                  for ex_i, input_ids in enumerate(tqdm(ds, desc=f"DSBD c={cmode} w={width}")):
                      cnt += 1
                      input_ids = input_ids.to(torch_device)
                      facts = _schema_facts_for_example(dataset_name, output_dataset, ex_i, spider_schema)
                      cm = _make_constraint_manager(cmode, tokenizer, facts)
                      try:
                        t = process_time_ns()
                        output, details = beam_speculative_sampling(
                          input_ids, small_model, large_model,
                          eos_token_id=tokenizer.eos_token_id,
                          pad_token_id=tokenizer.pad_token_id,
                          max_len=num_tokens,
                          gamma=gamma,
                          width=width,
                          num_beams=num_beams,
                          min_num_beams=min_w,
                          extra_sample_cnt=extra_sample_cnt,
                          expect_thres=w_thres,
                          top_k=top_k,
                          top_p=top_p,
                          random_seed=random_seed,
                          details=True,
                          constraint_manager=cm,
                        )
                        wall = process_time_ns() - t
                        total_time += wall
                        committed = len(output[0]) - input_ids.size(1)
                        total_token += committed
                        approx_time += details['approx_time']
                        target_time += details['target_time']
                        other_time += details['other_time']
                        total_acc_len += np.sum(details['acc_len'])
                        acc_rate.append(details['acc_rate'])
                        target_times += details['target_call_times']
                        approx_times += details['approx_call_times']
                        expect_cnt_list += details['expect_cnt_list']
                        compute_expect_time += details['compute_expect_time']
                        score = get_score(output, large_model, input_ids.size(1))
                        if score.isnan().any():
                          raise RuntimeError('score nan')
                        scores.append(score.item())
                        pred = _decode_pred(tokenizer, output, input_ids.size(1), dataset_name)
                        pred_seq.append(pred)
                        db_id = output_dataset[ex_i].split("[SQL]")[0] if dataset_name == "spider" else None
                        exec_acc = None
                        if dataset_name == "spider":
                            gt = output_dataset[ex_i].split("[SQL]")[1]
                            exec_acc = float(max(execution_accuracy(db_id, pred, gt), 0))
                        row = compute_goodput_row(
                            method="dsbd",
                            constraints=cmode,
                            example_idx=ex_i,
                            db_id=db_id,
                            pred_sql=pred if dataset_name == "spider" else "",
                            reference=output_dataset[ex_i] if dataset_name == "spider" else None,
                            committed_tokens=details.get("committed_tokens", committed),
                            tokens_proposed=details.get("tokens_proposed", committed),
                            tokens_accepted=details.get("tokens_accepted", 0),
                            tokens_constraint_rejected=details.get("tokens_constraint_rejected", 0),
                            xgrammar_rejects=details.get("xgrammar_rejects", 0),
                            z3_rejects=details.get("z3_rejects", 0),
                            wall_time_ns=details.get("wall_time_ns", wall),
                            xgrammar_time_ns=details.get("xgrammar_time_ns", 0),
                            z3_time_ns=details.get("z3_time_ns", 0),
                            acc_len_mean=float(np.mean(details["acc_len"])) if details.get("acc_len") else 0.0,
                            acc_rate=float(details.get("acc_rate") or 0.0),
                            exec_acc=exec_acc,
                            extra={"width": width, "gamma": gamma, "w_thres": w_thres,
                                   "min_w": min_w, "extra_sample_cnt": extra_sample_cnt},
                        )
                        _append_metrics_row(metrics_csv, row)
                        sum_proposed += row["tokens_proposed"]
                        sum_useful += row["n_useful"]
                        sum_wall += row["wall_time_s"]
                        if total_time / 1e9 > max_seconds:
                          log_both(f'terminated at {cnt}')
                          break
                      except Exception as e:
                          log_both(str(e))

                  t2 = time.time()
                  P.kill(); P.wait()
                  outputs = P.stdout.readlines()
                  fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_dsbd_{cmode}_{width}.pkl")
                  power_total = get_total_power(outputs, t1, t2, fname)
                  thr = (sum_proposed / sum_wall) if sum_wall > 0 else 0.0
                  gp = (sum_useful / sum_wall) if sum_wall > 0 else 0.0
                  log_both(f'\nDSBD (constraints={cmode}, gamma {gamma}, max_w {width}, min_w {min_w}, w_thres {w_thres}, extra {extra_sample_cnt}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/max(total_token,1)} s/token')
                  log_both(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                  log_both(f"average accepted len {total_acc_len/max(target_times,1)}, target call times {target_times}, acc rate {np.mean(acc_rate) if acc_rate else 0}, approx call times {approx_times}")
                  log_both(f"throughput={thr:.4f} tok/s, goodput={gp:.4f} useful-tok/s, proposed={sum_proposed}, useful={sum_useful}")
                  log_both(f'total power consumption: {power_total}')
                  em_score = None
                  cnt = len(pred_seq)
                  if em is not None and cnt > 0:
                      em_score = em(predictions=pred_seq[:cnt], references=output_dataset[:cnt])
                  log_both(f'em score = {em_score}')
                  if expect_cnt_list:
                      log_both(f'average expect cnt = {np.mean(expect_cnt_list)}')

        ######################################################################
        # Beam baseline (optional)
        ######################################################################
        if not skip_baselines and not use_fixed:
          for beams in [2,3,4]:
                max_beams = beams
                min_beams = beams
                total_time = 0
                total_token = 0
                scores = []
                pred_seq = []
                cnt = 0
                P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
                t1 = time.time()
                try:
                  for input_ids in tqdm(ds):
                    cnt += 1
                    input_ids = input_ids.to(torch_device)
                    t = process_time_ns()
                    output = random_width_beam_sampling(input_ids, large_model, num_tokens,
                        max_num_beams = max_beams, min_num_beams = min_beams,
                        eos_token_id = tokenizer.eos_token_id, 
                        top_k = top_k, top_p=top_p, pad_token_id = tokenizer.pad_token_id)
                    total_time += process_time_ns() - t
                    total_token += len(output[0])- input_ids.size(1)
                    score = get_score(output, large_model, input_ids.size(1))
                    scores.append(score.item())
                    pred_seq.append(_decode_pred(tokenizer, output, input_ids.size(1), dataset_name))
                    if total_time / 1e9 > max_seconds:
                        log_both(f'terminated at {cnt}')
                        break
                  t2 = time.time()
                  P.kill(); P.wait()
                  outputs = P.stdout.readlines()
                  fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_rwbd_{max_beams}_{min_beams}.pkl")
                  power_total = get_total_power(outputs, t1, t2, fname)
                  log_both(f'\nbeam decoding {(max_beams, min_beams)} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/max(total_token,1)} s/token')
                  if em is not None and pred_seq:
                      em_score = em(predictions=pred_seq, references=output_dataset[:cnt])
                      log_both(f'em score = {em_score}')
                except Exception as e:
                    print(e)
                    continue

    del large_model
    del small_model


if __name__ == "__main__":
    args = parse_arguments()
   
    
    evaluate(args.approx_model_name, args.target_model_name, 
            dataset_name = args.dataset,
            num_tokens=args.max_tokens, 
            top_k = args.top_k,
            top_p = args.top_p,
            max_seconds = args.max_seconds,
            log_file = args.log_file,
            random_seed = args.seed, 
            verbose=args.verbose,
            num_inputs = args.num_inputs,
            constraints=args.constraints,
            constraint_ablation=args.constraint_ablation,
            fixed_dsbd=args.fixed_dsbd,
            skip_baselines=args.skip_baselines,
            metrics_csv=args.metrics_csv,
            dsbd_width=args.dsbd_width,
            dsbd_gamma=args.dsbd_gamma,
            dsbd_w_thres=args.dsbd_w_thres,
            dsbd_min_w=args.dsbd_min_w,
            dsbd_extra_sample_cnt=args.dsbd_extra_sample_cnt,
            )
     
