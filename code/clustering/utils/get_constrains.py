import os
import json
import shutil

datasets_type = "small"

def read_cls_files(directory):
    all_data = []
    for filename in os.listdir(directory):
        if "CLs" in filename and filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            print(file_path)

            with open(file_path, 'r') as file:
                cl=[]
                for line in file:
                    data_dict = json.loads(line.strip())
                    cl.append(data_dict['number:'])
            all_data.append(cl)
    return all_data

#cl constrains
directory_path = f"code/constraint_generation/Cls_Result/{datasets_type}"
data = read_cls_files(directory_path)
print(len(data))
output_file = "code/clustering/constrains/small/CLS.json"

with open(output_file, 'w') as json_file:
    json.dump(data, json_file)

#ml constrains
ml_constrains_path = f"code/constraint_generation/Mls_Result/{datasets_type}/MLS.json"
ml_output_file = f"code/clustering/constrains/{datasets_type}/MLS.json"

src_file = ml_constrains_path
dst_file = ml_output_file
shutil.copy(src_file, dst_file)