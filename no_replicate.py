# json to jsonl
# import pandas as pd
# import json

# # List of paths to JSON files
# path_list = ["./source/his_intent.jsonl", "./data/问答对.json", "./data/data.json"]

# # List to store Pandas DataFrames
# dataframes = []

# # Read each JSON file into a Pandas DataFrame and append to the list
# for path in path_list:
#     with open(path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         df = pd.DataFrame(data)
#         dataframes.append(df)

# # Concatenate all DataFrames into one
# combined_df = pd.concat(dataframes, ignore_index=True)

# # Check for duplicates based on 'instruction' column
# deduplicated_df = combined_df.drop_duplicates(subset=[patterns])

# # Reset index to ensure continuous index after dropping duplicates
# deduplicated_df.reset_index(drop=True, inplace=True)

# # Shuffle the DataFrame if needed (optional)
# shuffled_df = deduplicated_df.sample(frac=1).reset_index(drop=True)

# # Save deduplicated data to a JSON file
# output_file = "./source/nintent.json"

# # Convert DataFrame to list of dictionaries (each row becomes a dictionary)
# output_data = shuffled_df.to_dict(orient='records')

# # Write the list of dictionaries to the JSON file
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(output_data, f, ensure_ascii=False, indent=2)

# print("Duplicates removed and saved successfully to", output_file)


import pandas as pd
import json

# Path to the JSONL input file
input_file = "./source/his_intent.jsonl"

# Path to the JSONL output file
output_file = "./source/nintent.jsonl"

# Dictionary to store unique patterns for each tag
unique_patterns = {}

# Read JSONL file line by line and deduplicate patterns
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line.strip())  
        tag = obj['tag']
        patterns = obj['patterns']

        if tag not in unique_patterns:
            unique_patterns[tag] = set() 

        unique_patterns[tag].update(patterns)

# Write deduplicated data back to JSONL format
with open(output_file, 'w', encoding='utf-8') as f:
    for tag, patterns in unique_patterns.items():
        obj = {
            "tag": tag,
            "patterns": list(patterns), 
            "responses": []  
        }
        json.dump(obj, f, ensure_ascii=False)
        f.write('\n')

print("Duplicates removed based on 'patterns' field and saved successfully to", output_file)

