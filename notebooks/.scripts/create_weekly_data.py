#!/usr/bin/env python3

import numpy as np
import shutil
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
    output_dir = os.path.join(DATA_ROOT, f"week_{target_week}")

    if (target_week == 1):
        # Week 1 is the same as initial data
        init_dir = os.path.join(DATA_ROOT, "initial_data")
        os.makedirs(output_dir, exist_ok=True)
        for index in range(8):
            init_func_dir = os.path.join(init_dir, f"function_{index+1}")
            out_func_dir = os.path.join(output_dir, f"function_{index+1}")
            os.makedirs(out_func_dir, exist_ok=True)

            shutil.copy2(os.path.join(init_func_dir, "initial_inputs.npy"), os.path.join(out_func_dir, "inputs.npy"))
            shutil.copy2(os.path.join(init_func_dir, "initial_outputs.npy"), os.path.join(out_func_dir, "outputs.npy"))
        
        return
    
    week_dir = os.path.join(DATA_ROOT, "updates", f"week_{target_week}")
    in_path = os.path.join(week_dir, "inputs.txt")
    out_path = os.path.join(week_dir, "outputs.txt")

    week_inputs = parse_inputs_txt(in_path)
    week_outputs = parse_outputs_txt(out_path)

    if not week_inputs or not week_outputs:
        print(f"Missing or corrupt files for week {target_week}.")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    for index in range(8):
        init_dir = os.path.join(DATA_ROOT, "initial_data", f"function_{index+1}")
        x_path = os.path.join(init_dir, "initial_inputs.npy")
        y_path = os.path.join(init_dir, "initial_outputs.npy")
        output_function_dir = os.path.join(output_dir, f"function_{index+1}")
        os.makedirs(output_function_dir, exist_ok=True)

        if os.path.exists(x_path) and os.path.exists(y_path):
            X_initial = np.load(x_path)
            Y_initial = np.load(y_path)

            try:
                num_batches = len(week_inputs) // 8
                X_new_list = [np.array(week_inputs[b * 8 + index]).reshape(1, -1) for b in range(num_batches)]
                Y_new_list = [week_outputs[b * 8 + index] for b in range(num_batches)]

                X_updated = np.vstack([X_initial] + X_new_list)
                Y_updated = np.append(Y_initial, Y_new_list)

                np.save(os.path.join(output_function_dir, f"inputs.npy"), X_updated)
                np.save(os.path.join(output_function_dir, f"outputs.npy"), Y_updated)

            except ValueError as e:
                print(f"Dimension mismatch at function {index+1}: {e}")
        else:
            print(f"Initial data for function {index+1} not found.")

    print(f"\nProcess complete. Files saved to: {output_dir}")

if __name__ == "__main__":
    main()