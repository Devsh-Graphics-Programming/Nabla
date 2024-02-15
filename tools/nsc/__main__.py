# Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.

# "nsc" target's test script, creates python environment for accessing 
# Nabla Python Framework module and runs the target's testing interface implementation

# Example of usage
# 1) python3 -m <reference to this module>
#	- runs all profiles found, goes through all batches found in a profile being processed
# 2) python3 -m <reference to this module> <profile path/index> [OPTIONAL] <batch index/range of batches> ... [OPTIONAL] <batch index/range of batches>
#	- runs given profile, if batch index (or range of batches) is given then it will process only the index of the batch (or the range)

import sys, os

testModulePath = os.path.dirname(__file__)
nblPythonFrameworkModulePath =  os.path.abspath(os.path.join(testModulePath, "../../tests/src"))

sys.path.append(nblPythonFrameworkModulePath) # relative path to Nabla's Python Framework module, relocatable

profile_count = sum(map(lambda arg: arg=="-c", sys.argv))
profile_path = os.path.join(testModulePath, ".profiles")
profile_args = list(map(lambda filename: os.path.join(profile_path, filename),\
    filter(lambda file: file.endswith(".json"), os.listdir(profile_path))))\
        if profile_count==0 else None
repository_path = os.path.abspath(os.path.join(testModulePath, "../../"))

from .test.test import main
main(None, profile_args, repository_path, True) # each test target implements its testing interface by overriding common one in Nabla Python Framework module
