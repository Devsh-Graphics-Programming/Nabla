import framework as nbl

def main(args = None, config_json_filepaths = None, nabla_dir = None, warnings = None):
    if config_json_filepaths is None:
        args, config_json_filepaths, nabla_dir, warnings = nbl.get_args()
    if nbl.ExpectedFileAsDependencyTest("Nabla DXC Tool", config_json_filepaths, nabla_dir, warnings).run():
        print("Test finished, passed")
        exit(0)
    else:
        print()
        exit(1)