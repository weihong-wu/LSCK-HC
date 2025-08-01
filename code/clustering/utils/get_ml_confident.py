import json
import random
import time
import re
import requests
from together import Together
from sklearn.metrics import adjusted_rand_score

datasets_name = 'banking77'
datasets_type = 'small'
prompt_name = "banking77_small_ml_prompt"

file_path = f'code/clustering/constrains/{datasets_type}/MLS.json'# ML path
data_path = f'code/datasets/{datasets_name}/{datasets_type}.jsonl'
ml_prompt_path = f"code/constraint_generation/prompts/{prompt_name}"
llm_api_key = "your api_key here"

if datasets_name == 'tweet':
    input_label = "text"
else:
    input_label = "input"

def load_prompt(prompt_file_path):
    prompt = ''
    with open(prompt_file_path, 'r') as file:
        for line in file:
            l = line.strip() + ' '
            prompt += l
    return prompt


def Llama_70B_instruct_turbo(content,temp):
    client = Together(api_key = llm_api_key)
    response = client.chat.completions.create(
        model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": content}],
        temperature=temp
    )
    return response.choices[0].message.content


def call_deepseek_api_with_backoff(content,temp=1.0, max_retries=999999999, initial_delay=2):
    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            return Llama_70B_instruct_turbo(content,temp)
        except Exception as e:
            error_msg = str(e)
            if 'Error code: 422' in error_msg:
                return '422'
            retries += 1
            print(f"API call failed ({retries}/{max_retries}), error: {e}, waiting {delay} seconds before retrying...")
            time.sleep(delay)
            delay += 2  # Increase wait time by 2 for each retry
            delay += random.uniform(0, 1)  # Add some randomness to avoid synchronized retries

    raise Exception(f"API call failed, maximum retries {max_retries} reached")



def are_nested_lists_equal(a, b):
    """
       Check if two nested lists are equal (regardless of order of sublists or elements within sublists).
       """
    sorted_a = [sorted(sublist) for sublist in a]
    sorted_b = [sorted(sublist) for sublist in b]

    sorted_a.sort()
    sorted_b.sort()
    return sorted_a == sorted_b

def ari_ll(list1, list2):

    unique_point_1 = set()
    unique_point_2 = set()
    unique_point_all = set()

    for l in list1:
        unique_point_1.update(l)
    for l in list2:
        unique_point_2.update(l)

    unique_point_all.update(unique_point_1)
    unique_point_all.update(unique_point_2)

    for l in unique_point_all:
        if l not in unique_point_1:
            list1.append([l])
        if l not in unique_point_2:
            list2.append([l])
    labels_1_dict = {}
    labels_2_dict = {}

    for label,ml in enumerate(list1):
        for m in ml:
            labels_1_dict[m] = label

    for label,ml in enumerate(list2):
        for m in ml:
            labels_2_dict[m] = label

    labels1 = []
    labels2 = []

    for i in unique_point_all:
        labels1.append(labels_1_dict[i])
        labels2.append(labels_2_dict[i])

    return adjusted_rand_score(labels1, labels2)

def load_json(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def binary_search_last_match_large(data, kpi_num, temp):
    left, right = 0, len(data) - 1
    final_out = 0
    print("the last ml sets in the grid: ", right)
    mid = 0
    mid_Flag = True
    while left <= right:
        if mid_Flag:
            mid_Flag = False
        else:
            mid = (left + right) // 2
        first_reply_list = None
        over_2_high_confidence_or_not = True

        ml_bank_prompt = load_prompt(ml_prompt_path)
        for i in range(len(data[mid])):
            if int(data[mid][i]) < len(data_to_dict):
                ml_bank_prompt = ml_bank_prompt + ' Target text ' + str(i + 1) + ': ' + \
                                 data_to_dict[int(data[mid][i])][input_label]
            else:
                ml_bank_prompt = None
                mid -= 1
                mid_Flag = True
                if right == mid + 1:
                    mid = right
                break

        if ml_bank_prompt != None:
            for time in range(10):
                try:
                    print(ml_bank_prompt)
                    res = call_deepseek_api_with_backoff(ml_bank_prompt, temp = temp)
                    kpi_num += 1
                    print('res:', res)
                    matches = re.findall(r'\[(.*?)\]', res)
                    if matches:
                        lists = [list(map(int, match.split(','))) for match in matches]

                        if first_reply_list is not None:
                            ari = ari_ll(lists, first_reply_list)
                            if ari < 0.9:
                                print(f"ari{ari}")
                                over_2_high_confidence_or_not = False
                                break
                        else:
                            first_reply_list = [i for i in lists]
                except:
                    continue

            print(f"mid = {mid}")
            print(over_2_high_confidence_or_not)
            if over_2_high_confidence_or_not:
                final_out = mid
                left = mid + 1
            else:
                right = mid - 1
    return final_out

# the pair one
def binary_search_last_match(data, kpi_num, temp, query_time = 5):
    final_out = 0
    left, right = 0, len(data) - 1
    print("the last ml sets in the grid: ", right)
    mid = 0
    mid_Flag = True
    while left <= right:
        if mid_Flag:
            mid_Flag = False
        else:
            mid = (left + right) // 2
        last_reply_list = None
        equal_2_high_confidence_or_not = False

        ml_bank_prompt = load_prompt(ml_prompt_path)
        for i in range(len(data[mid])):
            if int(data[mid][i]) < len(data_to_dict):
                ml_bank_prompt = ml_bank_prompt + ' Target text ' + str(i + 1) + ': ' + \
                                 data_to_dict[int(data[mid][i])][input_label]
            else: # some merged points in the sets
                ml_bank_prompt = None
                mid -= 1
                mid_Flag = True
                if right == mid + 1:
                    mid = right
                break

        if ml_bank_prompt != None:
            for _ in range(query_time):
                try:
                    print(ml_bank_prompt)
                    res = call_deepseek_api_with_backoff(ml_bank_prompt, temp = temp)
                    kpi_num += 1
                    matches = re.findall(r'\[(.*?)\]', res)
                    print(matches)
                    if matches:
                        lists = [list(map(int, match.split(','))) for match in matches]
                        if last_reply_list is not None:
                            equal_2_high_confidence_or_not = are_nested_lists_equal(lists, last_reply_list)
                            if equal_2_high_confidence_or_not is False:
                                print("False")
                                break
                        else:
                            last_reply_list = [i for i in lists]
                except:
                    continue

            print(f"mid = {mid}")
            print(equal_2_high_confidence_or_not)
            if equal_2_high_confidence_or_not:
                final_out = mid
                left = mid + 1
            else:
                right = mid - 1
    return final_out


#Main
data = load_json(data_path)
data_to_dict = {}
for num, dt in enumerate(data):
    data_to_dict[num] = dt

with open(file_path, 'r') as f:
    data = json.load(f)

def pair_confi():
    kpi_num = 0
    full_sub_dict = {}
    pair_Flag = True
    set_Flag = True

    output = {}
    for outer_key, sub_dict in data.items():
        result_pair = -1
        result_set = -1
        print(sub_dict.items())
        flat_values_pair = []
        flat_values_set = []
        for inner_key, value_list in sub_dict.items():
            full_sub_dict[inner_key] = value_list
            if len(value_list) == 2:
                flat_values_pair.append(value_list)
            elif len(value_list) > 2:
                flat_values_set.append(value_list)

        if pair_Flag:
            print("pair: ")
            print(flat_values_pair)
            result_pair = binary_search_last_match(flat_values_pair, kpi_num, 1.3, 5)
            print("final: 1) outer_key", result_pair)
            print("kpi_num:,", kpi_num)
            print(result_pair)
            if result_pair < len(flat_values_pair) - 1:
                pair_Flag = False
        if set_Flag:
            print(len(flat_values_set))
            print("set: ")
            print(flat_values_set)
            result_set = binary_search_last_match_large(flat_values_set,  kpi_num, 1)
            print(result_set)
            if result_set < len(flat_values_set) - 1:
                set_Flag = False
        output[int(outer_key)] = [result_pair+1,result_set+1]
    print(output)
    return output

conf = pair_confi()

with open('./ml_confidence.json', 'r', encoding='utf-8') as f:
    data_confidence = json.load(f)

data_confidence[f'{datasets_name}_{datasets_type}'] = conf

data_confidence_path = "./ml_confidence.json"

with open(data_confidence_path, 'w') as json_file:
    json.dump(data_confidence, json_file,ensure_ascii=False, indent=2)