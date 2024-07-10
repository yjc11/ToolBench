"""
I1 Generation:

Use sliding window to sample apis from a tool.
Window size:10
Step size:3

! Change paths before running the program.

Since adding tool name to api name for generated relevant apis increases hallucination, tool name is added afterwards.
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
from tqdm import tqdm

# api list path
input_path = r"/data/private/wanghuadong/liangshihao/ToolBench/BMTools/rapidapi/free_apis/jsons_filtered_pipeline_paid_subscribed_white_list"
# target output path
output_path = r"/data/private/yanlan/QueriesChatgpt0712_singletool_paid_slidingwindow6"

# singel tool seed path
import glob
txt_directory = '/data/private/yanlan/QueriesChatgpt0621-5Sampled_Plaintext'
# Construct seed list
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
    # Replace single quotes with double quotes for JSON format
    fixed_string = long_string.replace("['", '[\"').replace("']", '\"]')

    # Parse the string as JSON
    queries_json = json.loads(fixed_string)
    # Initialize an empty list for storing processed queries
    processed_queries = []

    # Process each query
    for query_obj in queries_json:
        # For each query object, get rid of the leading "Query"
        new_key = re.sub(r'^Query\d+: ', '', list(query_obj.keys())[0].strip())
        # Fetch related_apis
        related_apis = query_obj['related_apis']
        # Add the processed query to the list
        processed_queries.append({"Query": query_obj[new_key], 'related_apis': related_apis})

    return processed_queries


def process_file(paths):
    try:
        input_path, output_folder_path = paths
        os.makedirs(output_folder_path, exist_ok=True) # Create the output folder if it doesn't exist
        output_path = os.path.join(output_folder_path, os.path.basename(input_path))

        if os.path.exists(output_path):
            return "File already exists"
        
        pattern = r'/(?P<word>[^/]+)/[^/]+$'
        match = re.search(pattern, input_path)
        if match:
            category = match.group('word')
        else:
            print("No match category found.")

        try:
            with open(input_path, "r",encoding='utf-8') as f:
                data = json.load(f)
        except:
            return "File is not a valid json file at {}".format(input_path)
        
        if len(data["api_list"]) <= 1:
            return "Empty or Only one API {}".format(output_path)
        
        
        prompt = """You will be provided with a tool, its description, all of the tool's available API functions, the descriptions of these API functions, and the parameters required for each API function. Your task involves creating 10 varied, innovative, and detailed user queries that employ multiple API functions of a tool. For instance, if the tool 'climate news' has three API calls - 'get_all_climate_change_news', 'look_up_climate_today', and 'historical_climate', your query should articulate something akin to: first, determine today's weather, then verify how often it rains in Ohio in September, and finally, find news about climate change to help me understand whether the climate will change anytime soon. This query exemplifies how to utilize all API calls of 'climate news'. A query that only uses one API call will not be accepted. Additionally, you must incorporate the input parameters required for each API call. To achieve this, generate random information for required parameters such as IP address, location, coordinates, etc. For instance, don't merely say 'an address', provide the exact road and district names. Don't just mention 'a product', specify wearables, milk, a blue blanket, a pan, etc. Don't refer to 'my company', invent a company name instead. The first seven of the ten queries should be very specific. Each single query should combine all API call usages in different ways and include the necessary parameters. Note thqt you shouldn't ask 'which API to use', rather, simply state your needs that can be addressed by these APIs. You should also avoid asking for the input parameters required by the API call, but instead directly provide the parameter in your query. The final three queries should be complex and lengthy, describing a complicated scenario where all the API calls can be utilized to provide assistance within a single query. You should first think about possible related API combinations, then give your query. Related_apis are apis that can be used for a give query; those related apis have to strictly come from the provided api names. For each query, there should be multiple related_apis; for different queries, overlap of related apis should be as little as possible. Deliver your response in this format: [{Query1: ......, 'related_apis':[api1, api2, api3...]},{Query2: ......, 'related_apis':[api4, api5, api6...]},{Query3: ......, 'related_apis':[api1, api7, api9...]}, ...] \n"""

        output_dict = {}
        output_dict["tool_name"] = data["tool_name"]
        output_dict["tool_description"] = data["tool_description"]
        

        tmp_data = data.copy()

        tmp_data.pop("product_id", None)
        tmp_data.pop("home_url", None)
        tmp_data.pop("title", None)
        tmp_data.pop("pricing", None)
        tmp_data.pop("score", None)
        tmp_data.pop("host", None)
        tmp_data.pop("tool_name", None)

        all_apis = []

        for api in data["api_list"]:
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
            tmp_api["category_name"] = category  # Adding the category_name key
            
            all_apis.append(tmp_api)

        # Define window size and step size
        window_size = 10
        step_size = 3

        # Loop using sliding window
        for i in range(0, len(all_apis), step_size):

            # Extract the APIs in the current window
            window_apis = all_apis[i: i + window_size]

            # Stop when the window doesn't contain 10 APIs, and the total number is greater than 10
            if (len(window_apis) < window_size) and (len(all_apis) > window_size):
                break
            if (len(window_apis) < window_size) and (len(all_apis) <= window_size) and (i != 0):
                break

            # Populate tmp_data with the APIs in the current window
            tmp_data["api_list"] = window_apis

            output_dict["api_list"] = []
            # output_dict["api_names_list"] = []
            output_dict["query_results"] = []

            for api in window_apis:
                if "template_response" in api:
                    output_dict["api_list"].append({"category_name": category,"tool_name": data["tool_name"],"api_name": api["name"], "api_description": api["description"], "required_parameters": api["required_parameters"], "optional_parameters": api["optional_parameters"], "method": api['method'], "template_response": api["template_response"]})
                else:
                    output_dict["api_list"].append({"category_name": category,"tool_name": data["tool_name"],"api_name": api["name"], "api_description": api["description"], "required_parameters": api["required_parameters"], "optional_parameters": api["optional_parameters"], "method": api['method']})

            openai = OpenAI({"model": "gpt-3.5-turbo-16k", "temperature": 0.6, "max_tokens": 2048, "top_p": 1, "best_of": 1})

            pro = "\nPlease produce ten queries in line with the given requirements and inputs. These ten queries should display a diverse range of sentence structures: some queries should be in the form of imperative sentences, others declarative, and yet others, interrogative. Equally, they should encompass a variety of tones, with some being polite, others straightforward. Ensure they vary in length and contain a wide range of subjects: myself, my friends, family, and company. Aim to include a number of engaging queries as long as they relate to API calls. Keep in mind that for each query, invoking just one API won't suffice; each query should call upon two to five APIs. However, try to avoid explicitly specifying which API to employ in the query. Each query should consist of a minimum of thirty words."

            # Sample three items from examples (assuming you have a list called examples)
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

            # Appending logic
            if output_dict["query_results"]:
                output_dict["query_results"][0]["queries"].extend(queries_list)
            else:
                output_dict["query_results"].append({"queries": queries_list})

            
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=4)

        return "Finish processing {}".format(output_path)
    
    except:
        return "Error when processing {}".format(output_path)
    
def get_all_files(input_path, output_path):
    all_files = []
    for folder in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder)
        output_folder_path = os.path.join(output_path, folder)
        for file in os.listdir(folder_path):
            all_files.append((os.path.join(folder_path, file), output_folder_path))
    return all_files

if __name__ == "__main__":
    all_files = get_all_files(input_path, output_path)

    with multiprocessing.Pool(processes=50) as pool:
        for message in tqdm(pool.imap(process_file, all_files), total=len(all_files)):
            try:
                print(message)
            except:
                continue




