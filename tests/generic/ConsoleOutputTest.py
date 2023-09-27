import shutil
from enum import Enum
import sys
import os
from pathlib import *

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from CITest import *

class ConsoleOutputTest(CITest):

    def _get_input_lines(self):
        return [self.input]

    def __init__(self, 
                test_name: str,
                executable_filepath: str, 
                input_filepath: str, 
                nabla_repo_root_dir: str, 
                executable_args = "",
                print_warnings = True,
                ):
        super().__init__(test_name, executable_filepath, input_filepath, nabla_repo_root_dir, print_warnings)
        self.executable_args = executable_args

   
    # run a test for a single line of input for pathtracer
    def _impl_run_single(self, input_args:str) -> dict:
        with open(self.input, "r") as file:
            expected = file.read().strip().replace('\r','') 
            #deleting \r from both to make sure neither are CRLF 

        console_output = subprocess.run([self.executable, input_args] ,capture_output=True).stdout.decode().strip().replace('\r','')
        if(console_output != expected):
            return {
                'status': 'failed',
                'status_color': 'red',
            }
        else:
            return {
                'status': 'passed',
                'status_color': 'green',
            }
    
    


def run_all_tests(args):
    # test public scenes
    CI_PASS_STATUS = ConsoleOutputTest(*args).run()

    # check if both were successful
    if not (CI_PASS_STATUS):
        print('CI failed, output did not match the file')
        exit(-2)
    print('CI done')
    exit()


if __name__ == '__main__':
    run_all_tests(sys.argv[1::])
