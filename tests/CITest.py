import datetime
import os
import re
import subprocess
import filecmp
import json
from pathlib import *

def get_git_revision_hash(repo_dir, branch = "ditt") -> str:
    return subprocess.check_output(f'git -C "{repo_dir}" rev-parse origin/{branch}').decode('ascii').strip()

class CITest:

    # here are the methods that need to be implemented in a derived class
    # optional override
    def _impl_run_dummy_case(self):
        return True

    # must override 
    def _impl_run_single(self, input_args) -> dict:
        pass



    def _get_input_lines(self) :
        with open(self.input.absolute()) as file:
            inputLines = file.readlines()
        # filter out commented lines
        inputLines = list(filter(lambda line: not line.startswith(';'), inputLines))
        return inputLines


    def _cmp_files(self, fileA, fileB, cmpByteByByte = False, cmpSavedHash = False, cmpSavedHashLocation = "" ):
        if cmpByteByByte:
            return filecmp.cmp(fileA, fileB)
        #compare git hashes
        executor1 = f'git hash-object {fileA}'
        executor2 = f'git hash-object {fileB}'
        hashA = subprocess.run(executor1, capture_output=True).stdout.decode().strip()
        hashB = subprocess.run(executor2, capture_output=True).stdout.decode().strip()
        res = hashA == hashB
        if res:
            if cmpSavedHash:
                if Path(cmpSavedHashLocation).is_file():
                    with open(cmpSavedHashLocation, "r") as f:
                        res = res and hashA == f.readline().strip()
                elif self.print_warnings:
                    print(f"[WARNING] could not compare git hash of file '{hashA}' and '{hashB}' with a hash stored in a text file {cmpSavedHashLocation}. Text file storing hash does not exist")
        elif self.print_warnings:
            print(f"[WARNING] files have different git commit hashes: '{fileA}', '{fileB}'")
        return res


    def __init__(self, 
                test_name : str,
                executable_filepath : str,
                input_filepath : str,
                nabla_repo_root_dir : str,
                print_warnings = True
                ):
        self.test_name = test_name
        self.alphanumeric_only_test_name = re.sub(r"[^A-z0-9]+", "_", test_name).strip("_") #remove all non alphanumeric characters 
        self.executable = Path(executable_filepath)
        self.input = Path(input_filepath)
        self.print_warnings = print_warnings
        self.nabla_repo_root_dir = nabla_repo_root_dir
        self._validate_filepaths()

   
    def _change_working_dir(self):
        os.chdir(self.executable.parent.absolute()) 


    def _validate_filepaths(self):
        if not self.executable.exists():
            if self.print_warnings:
                print(f'[WARNING] Executable at path "{self.executable}" does not exist')
            return False
        if not self.input.exists():
            if self.print_warnings:
                print(f'[WARNING] Input file at path "{self.input}" does not exist')
            return False
        return True
    

    def _save_json(self, jsonFilename, dict):
        jsonobj = json.dumps(dict, indent = 2)
        file = open(jsonFilename, "w")
        file.write(jsonobj)
        file.close()


    def run(self, inputParamList):
        self._change_working_dir()
        summary = { 
            "commit": self.__get_commit_data(),
            "datetime": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        }
        test_results = []
        ci_pass_status = True
        if not self._impl_run_dummy_case():
            ci_pass_status = False
        else:
            input_lines = self._get_input_lines()
            summary["num_of_tests"] = len(input_lines)
            failures = 0
            testnum = 0
            for line in input_lines:
                try:
                    testnum = testnum + 1
                    result = self._impl_run_single(line)
                    result["index"] = testnum
                    self._save_json(f"result_{self.alphanumeric_only_test_name}_{testnum}.json", result)
                    is_failure = not result["pass_status"]
                    if is_failure:
                        failures = failures + 1
                    test_results.append(result)
                except Exception as ex:
                    print(f"[ERROR] Critical exception occured during testing input {line}: {str(ex)}")
                    ci_pass_status = False
                    summary["critical_errors"] = f"{line}: {str(ex)}"
                    break
        summary["failure_count"] = failures 
        summary["pass_status"] = ci_pass_status 
        summary["results"] = test_results
        self._save_json(f"summary_{self.alphanumeric_only_test_name}.json",summary)
        return ci_pass_status


    def __get_commit_data(self):
        lines = subprocess.check_output(f'git -C "{self.nabla_repo_root_dir}" show').decode('ascii').strip().splitlines()
        return {
            'hash' : re.search(r"(?<=commit )\w+", lines[0]).group(),
            'author': lines[1],
            'date': lines[2],
            'name': lines[4].strip()
        }
