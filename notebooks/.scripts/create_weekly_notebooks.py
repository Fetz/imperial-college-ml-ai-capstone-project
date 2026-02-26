#!/usr/bin/env python3
import os
import shutil

def main():
    try:
        target_week = int(input("Create data for what week? "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    if (target_week < 1):
        print("Invalid input. Please enter a week bigger than 1.")
        return
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    NOTEBOOKS_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
    previous_week = target_week - 1

    for index in range(8):
        previous_week_file = os.path.join(NOTEBOOKS_ROOT, f"week_{previous_week}_function_{index+1}.ipynb")
        target_week_file = os.path.join(NOTEBOOKS_ROOT, f"week_{target_week}_function_{index+1}.ipynb")

        shutil.copy2(previous_week_file, target_week_file)

if __name__ == "__main__":
    main()