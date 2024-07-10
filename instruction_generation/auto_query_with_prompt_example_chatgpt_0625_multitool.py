"""
I2 Generation:

Use category to acquire possible tool combination.
For each generation 2-5 tools are sampled from a category, and for each sampled tool randomly retrieve at most three apis of the tool to form the whole api set.

For each category, the number of times of selecting tools is proportional to tools in the category.

! Change paths before running the program.
"""

from typing import List, Optional
import requests
import json
import logging
import os
import random
import requests
import re
import multiprocessing
import glob
from tqdm import tqdm

# api path
input_path = r"/data/private/wanghuadong/liangshihao/ToolBench/BMTools/rapidapi/free_apis/jsons_filtered_pipeline_subscribed_white_list"
# target output path
output_path = r"/data/private/yanlan/QueriesChatgpt0705-multitool3"
# multitool(multicategory) seed path
txt_directory = '/data/private/yanlan/QueriesChatgpt0625-1multitool_multiSampled_Plaintext'

# construct seed list, later choose 3 from the list for each generation
examples = []
for txt_path in glob.glob(os.path.join(txt_directory, '*.txt')):
    with open(txt_path, 'r') as file:
        content = file.read()
    examples.append(content)

class OpenAI(object):
    def __init__(self, api_info):
        """
        :param api_info: a dict containing params for api generates like {"temp": 0.9}
        """
        self.url = "your ChatGPT server"
        self.model = api_info["model"] 
        self.temperature = api_info["temperature"]
        self.max_tokens = api_info["max_tokens"]
        self.top_p = api_info["top_p"]
        self.best_of = api_info["best_of"]

    def get_response(self, prompt):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are QueryGPT, a helpful assistant who can strictly follow my instructions to generate diverse real queries."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "best_of": self.best_of,
            "stop": None
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(self.url, json=payload, headers=headers)
        return response

def process_json_dict(d):
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, list):
            new_dict[key] = process_json_dict(value[0]) if value and isinstance(value[0], dict) else (value[0] if value else None)
        elif isinstance(value, dict):
            new_dict[key] = process_json_dict(value)
        else:
            new_dict[key] = value
    return new_dict

def dict_schema(d: dict):
    """
    Given a dictionary, return a dictionary of the same structure but with the values replaced by their types.
    """
    schema = {}
    for key, value in d.items():
        if isinstance(value, dict):
            schema[key] = dict_schema(value)
        elif isinstance(value, list):
            if not value:  # check if the list is empty
                schema[key] = "empty list"
            elif isinstance(value[0], dict):
                inner_schema = dict_schema(value[0])
                inner_schema['_list_length'] = len(value)
                schema[key] = [inner_schema]
            else:
                schema[key] = ["list of " + type(value[0]).__name__ + " with length " + str(len(value))]
        else:
            schema[key] = type(value).__name__
    return schema

def process_numbered_strings(long_string):
    queries_json = json.loads(long_string)
    processed_queries = []
    for query_obj in queries_json:
        new_key = re.sub(r'^Query\d+: ', '', list(query_obj.keys())[0].strip())
        related_apis = query_obj['related_apis']
        processed_queries.append({"Query": query_obj[new_key], 'related_apis': related_apis})
    return processed_queries

def process_file(file_info):
    try:
        input_paths, output_folder_path = file_info
        os.makedirs(output_folder_path, exist_ok=True)

        # Extract category from the first input path
        category = os.path.basename(os.path.dirname(input_paths[0]))
        
        # Combine the tool names to create the output file name
        output_filename = '$'.join(os.path.splitext(os.path.basename(path))[0] for path in input_paths) + '.json'
        
        # Combine output_folder_path, category, and output_filename to create the final output path
        final_output_path = os.path.join(output_folder_path, output_filename)
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

        output_dict = {
            "tool_name": [],
            "tool_description": [],
            "api_list": [],
            "query_results": []
        }
        
        tmp_data = {
            "api_list": []
        }
        
        for input_path in input_paths:
            try:
                with open(input_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
            except:
                return f"File is not a valid json file at {input_path}"

            output_dict["tool_name"].append(data["tool_name"])
            output_dict["tool_description"].append(data["tool_description"])

            # Shuffle the api list
            api_list = data["api_list"]
            random.shuffle(api_list)

            # Get a maximum of 3 APIs for each tool
            api_list = api_list[:3]

            for api in api_list:
                tmp_api = api.copy()
                max_length = 1000
                
                if "test_endpoint" in tmp_api:
                    if isinstance(tmp_api["test_endpoint"], list):
                        if len(tmp_api["test_endpoint"]) == 0:
                            tmp_api["template_response"] = {}
                        elif isinstance(tmp_api["test_endpoint"][0], dict):
                            tmp_api["template_response"] = dict_schema(tmp_api["test_endpoint"][0].copy())
                        else:
                            tmp_api["template_response"] = {}
                        if len(json.dumps(tmp_api["template_response"])) > max_length:
                            tmp_api["template_response"] = json.dumps(tmp_api["template_response"])[:max_length]
                    elif isinstance(tmp_api["test_endpoint"], dict):
                        tmp_api["template_response"] = dict_schema(tmp_api["test_endpoint"].copy())
                        if len(json.dumps(tmp_api["template_response"])) > max_length:
                            tmp_api["template_response"] = json.dumps(tmp_api["template_response"])[:max_length]
                    del tmp_api["test_endpoint"]
                tmp_api.pop("code", None)
                tmp_api.pop("convert_code", None)
                tmp_api["tool_name"] = data["tool_name"]
                tmp_data["api_list"].append(tmp_api)
                
                if "template_response" in tmp_api:
                    output_dict["api_list"].append({
                        "category_name": category,
                        "tool_name": data["tool_name"],
                        "api_name": api["name"],
                        "api_description": api["description"],
                        "required_parameters": api["required_parameters"],
                        "optional_parameters": api["optional_parameters"],
                        "method": api['method'],
                        "template_response": tmp_api["template_response"]
                    })
                else:
                    output_dict["api_list"].append({
                        "category_name": category,
                        "tool_name": data["tool_name"],
                        "api_name": api["name"],
                        "api_description": api["description"],
                        "required_parameters": api["required_parameters"],
                        "optional_parameters": api["optional_parameters"],
                        "method": api['method']
                    })
                
        openai = OpenAI({"model": "gpt-3.5-turbo-16k", "temperature": 0.6, "max_tokens": 2048, "top_p": 1, "best_of": 1})

        prompt = """You will be provided with several tools, tool descriptions, all of each tool's available API functions, the descriptions of these API functions, and the parameters required for each API function. Your task involves creating 10 varied, innovative, and detailed user queries that employ API functions of multiple tools. For instance, given three tools 'nba_news' 'cat-facts' 'hotels': 'nba_news' has API functions "Get individual NBA source news" and "Get all NBA news", "cat-facts" has API functions "Get all facts about cat" and "Get a random fact about cats", 'hotels' has API functions "properties/get-details (Deprecated)", "properties/list (Deprecated)" and "locations/v3/search". Your query should articulate something akin to: "I want to name my newborn cat after Kobe and host a party to celebrate its birth. Get me some cat facts and nba news to gather inspirations for the cat name. Also, find a proper hotel around my house in Houston Downtown for the party." This query exemplifies how to utilize API calls of all the given tools. A query that uses API calls of onlyy one tool will not be accepted. Additionally, you must incorporate the input parameters required for each API call. To achieve this, generate random information for required parameters such as IP address, location, coordinates, etc. For instance, don't merely say 'an address', provide the exact road and district names. Don't just mention 'a product', specify wearables, milk, a blue blanket, a pan, etc. Don't refer to 'my company', invent a company name instead. The first seven of the ten queries should be very specific. Each single query should combine API calls of different tools in various ways and include the necessary parameters. Note that you shouldn't ask 'which API to use', rather, simply state your needs that can be addressed by these APIs. You should also avoid asking for the input parameters required by the API call, but instead directly provide the parameters in your query. The final three queries should be complex and lengthy, describing a complicated scenario where all the provided API calls can be utilized to provide assistance within a single query. You should first think about possible related API combinations, then give your query. Related_apis are apis that can be used for a give query; those related apis have to strictly come from the provided api names. For each query, there should be multiple related_apis; for different queries, overlap of related apis should be as little as possible. Deliver your response in this format: [{Query1: ......, 'related_apis':[[<tool name>, <api name>], [<tool name>, <api name>], [<tool name>, <api name>]...]},{Query2: ......, 'related_apis':[[<tool name>, <api name>], [<tool name>, <api name>], [<tool name>, <api name>]...]},{Query3: ......, 'related_apis':[[<tool name>, <api name>], [<tool name>, <api name>], [<tool name>, <api name>]...]}, ...] \n"""

        pro = "\nPlease produce ten queries in line with the given requirements and inputs. These ten queries should display a diverse range of sentence structures: some queries should be in the form of imperative sentences, others declarative, and yet others, interrogative. Equally, they should encompass a variety of tones, with some being polite, others straightforward. Ensure they vary in length and contain a wide range of subjects: myself, my friends, family, and company. Aim to include a number of engaging queries as long as they relate to API calls. Keep in mind that for each query, invoking just one API won't suffice; each query should call upon two to five APIs. However, try to avoid explicitly specifying which API to employ in the query. Each query should consist of a minimum of thirty words."

        sampled_items = random.sample(examples, 3)
        pro_example = "\n".join(sampled_items)
        prompt_ = prompt + pro_example +  "\nThese are only examples to show you how to write the query. Do not use apis listed in the above examples, but rather, use the ones listed below in the INPUT.\nINPUT:\n" + json.dumps(tmp_data, indent=4) + pro + "\nOUTPUT:\n"

        response = openai.get_response(prompt_[:80000])
        try:
            queries = json.loads(response.text)["choices"][0]['message']['content']
        except:
            try:
                response = openai.get_response(prompt_[:70000])
                queries = json.loads(response.text)["choices"][0]['message']['content']
            except:
                return "Error when calling openai {}".format(output_path)
        queries_list = process_numbered_strings(queries)
        output_dict["query_results"].append({"queries": queries_list})

        with open(final_output_path, "w") as f:
            json.dump(output_dict, f, indent=4)

        return "Finish processing {}".format(output_path)
    except:
        return "Error when processing {}".format(output_path)
    
def get_all_files(input_path, output_path):
    all_files = []
    for folder in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder)
        output_folder_path = os.path.join(output_path, folder)
        json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
        k = (len(json_files) // 5) * 20 + 4
        for _ in range(k):
            n = random.randint(2, 5)
            sampled_files = random.sample(json_files, min(n, len(json_files)))
            json_file_paths = [os.path.join(folder_path, file) for file in sampled_files]
            all_files.append((json_file_paths, output_folder_path))
    return all_files


if __name__ == "__main__":
    all_files = get_all_files(input_path, output_path)

    with multiprocessing.Pool(processes=25) as pool:
        for message in tqdm(pool.imap(process_file, all_files), total=len(all_files)):
            try:
                print(message)
            except:
                continue




