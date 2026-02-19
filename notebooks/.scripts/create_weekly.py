#!/usr/bin/env python3

import numpy as np
import os
import re

def parse_inputs_txt(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            
            # Split the file by 'array(['
            # The first part [0] will be the headers like '[' so we skip it
            parts = content.split('array([')[1:]
            
            parsed_data = []
            for p in parts:
                # The data for this array ends at '])'
                # We split at '])' and take the first half
                inner_content = p.split('])')[0]
                
                # Extract all numbers (handles decimals, negatives, and scientific notation)
                nums = re.findall(r"[-+]?\d*\.\d+|\d+[eE][-+]?\d*|\d+", inner_content)
                parsed_data.append([float(n) for n in nums])
            
            return parsed_data # This will return exactly as many arrays as it finds
    except Exception as e:
        print(f"Error parsing inputs {filepath}: {e}")
        return None

def parse_outputs_txt(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            # Find values inside np.float64(...)
            blocks = re.findall(r"np\.float64\((.*?)\)", content)
            return [float(b) for b in blocks]
    except Exception as e:
        print(f"Error parsing outputs {filepath}: {e}")
        return None

def main():    
    try:
        target_week = int(input("Create data for what week? "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
    
    functions_data = [[[], []] for _ in range(8)]

    for current_week in range(1, target_week + 1):
        week_dir = os.path.join(DATA_ROOT, "updates", f"week_{current_week}")
        in_path = os.path.join(week_dir, "inputs.txt")
        out_path = os.path.join(week_dir, "outputs.txt")

        week_inputs = parse_inputs_txt(in_path)
        week_outputs = parse_outputs_txt(out_path)

        if week_inputs and week_outputs:
            for i in range(8):
                functions_data[i][0].append(week_inputs[i])
                functions_data[i][1].append(week_outputs[i])
        else:
            print(f"Skipping week {current_week} due to missing or corrupt files.")

    output_dir = os.path.join(DATA_ROOT, f"week_{target_week}")
    os.makedirs(output_dir, exist_ok=True)

    for index in range(8):
        init_dir = os.path.join(DATA_ROOT, "initial_data", f"function_{index+1}")
        x_path = os.path.join(init_dir, "initial_inputs.npy")
        y_path = os.path.join(init_dir, "initial_outputs.npy")
        output_function_dir = os.path.join(output_dir, f"function_{index+1}")
        os.makedirs(output_function_dir, exist_ok=True)

        if os.path.exists(x_path) and os.path.exists(y_path):
            X_updated = np.load(x_path)
            Y_updated = np.load(y_path)

            X_new_list = np.array(functions_data[index][0])
            Y_new_list = np.array(functions_data[index][1])

            try:
                for X_item in X_new_list:
                    X_item = np.array(X_item).reshape(1, -1)
                    X_updated = np.vstack((X_updated, X_item))
                
                for Y_item in Y_new_list:
                    Y_updated = np.append(Y_updated, Y_item)

                np.save(os.path.join(output_function_dir, f"inputs.npy"), X_updated)
                np.save(os.path.join(output_function_dir, f"outputs.npy"), Y_updated)
                
            except ValueError as e:
                print(f"Dimension mismatch at function {index}: {e}")
        else:
            print(f"Initial data for function {index+1} not found.")

    print(f"\nProcess complete. Files saved to: {output_dir}")

if __name__ == "__main__":
    main()