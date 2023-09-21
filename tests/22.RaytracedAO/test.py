import shutil
from enum import Enum
import sys
import os
from pathlib import *

current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
 
# now we can import the module in the parent
# directory.
from CITest import *

EPSILON = "0.00001"         
NBL_REF_LDS_CACHE_FILENAME = 'LowDiscrepancySequenceCache.bin'
PUT_REFERENCES_IN_FOLDERS = True
PUT_DIFFERENCES_IN_FOLDERS = True
PUT_RENDERS_IN_FOLDERS = True


class ErrorThresholdType(Enum):
    ABSOLUTE = 1
    RELATIVE_TO_RESOLUTION = 2

class RendersTest(CITest):
    # self.executable 
    # self.input 
    # self.print_warnings
    # self.nabla_repo_root_dir
    
    # render tests need working directories to exist
    def _validate_filepaths(self):
        if not super()._validate_filepaths():
            return False
        if not self.data_renders_abs_dir.exists():
            os.makedirs(self.data_renders_abs_dir)

        if not self.data_references_abs_dir.exists():
            os.makedirs(self.data_references_abs_dir)
                
        if not self.data_diffs_abs_dir.exists():
            os.makedirs(self.data_diffs_abs_dir)
        
        if not self.references_repo_dir.exists():
            if self.print_warnings:
                print(f"[WARNING] References repository dir does not exist {self.references_repo_dir}")
            return False

    # constructor
    def __init__(self, 
                test_name: str,
                executable_filepath: str, 
                input_filepath: str, 
                nabla_repo_root_dir: str, 
                image_magick_exe: str,
                references_repo_dir: str,
                profile,
                data_dir = "/examples_tests/22.RaytracedAO/bin",
                renders_dir_name = "renders",
                references_dir_name = "references",
                diff_images_dir_name = "diff_images",
                error_threshold_type = ErrorThresholdType.ABSOLUTE,
                error_threshold_value = 0.05,
                allowed_error_pixel_count = 100.0,
                ssim_error_threshold_value = 0.0001,
                print_warnings = True
                ):
        super().__init__(test_name, executable_filepath, input_filepath, nabla_repo_root_dir, print_warnings)
        self.references_repo_dir = Path(references_repo_dir)
        self.data_dir = Path(str(self.nabla_repo_root_dir.absolute()) + data_dir)
        self.data_renders_rel_dir = "/" + renders_dir_name + "/" + profile
        self.data_renders_abs_dir = Path(str(self.data_dir.absolute()) + self.data_renders_rel_dir)
        self.data_references_rel_dir = "/" + references_dir_name + "/" + profile
        self.data_references_abs_dir = Path(str(self.data_dir.absolute())+ self.data_references_rel_dir)
        self.data_diffs_rel_dir = "/" + diff_images_dir_name + "/" + profile
        self.data_diffs_abs_dir = Path(str(self.data_dir.absolute())+ self.data_diffs_rel_dir)
        self.image_magick_exe = image_magick_exe
        self.error_threshold_type = error_threshold_type
        self.error_threshold_value = error_threshold_value
        self.allowed_error_pixel_count = allowed_error_pixel_count
        self.ssim_error_threshold_value = float(ssim_error_threshold_value)
        self.cout_json_regex = re.compile(r"(?<=\[JSON\] )(.+[\n\r]*)+(?=[\n\r]*\[ENDJSON\])")

    def __get_lds_hash(self):
        executor = f'git hash-object {NBL_REF_LDS_CACHE_FILENAME}'
        return subprocess.run(executor, capture_output=True).stdout.decode().strip()
    
    def __get_ref_repo_hash(self):
        return get_git_revision_hash(self.references_repo_dir)
        
    def _impl_run_dummy_case(self):
        reference_lds_cache_exists = bool(Path(str(self.data_references_abs_dir) + '/' + NBL_REF_LDS_CACHE_FILENAME).exists())

        generatedReferenceCache = NBL_REF_LDS_CACHE_FILENAME
        destinationReferenceCache = str(self.data_references_abs_dir) + '/' + NBL_REF_LDS_CACHE_FILENAME
        sceneDummyRender = '"../ci/dummy_4096spp_128depth.xml"'
        executor = str(self.executable.absolute()) + ' -SCENE=' + sceneDummyRender + ' -PROCESS_SENSORS RenderAllThenTerminate 0'
        subprocess.run(executor, capture_output=True)

        if not reference_lds_cache_exists:
            if self.print_warnings:
                print(f"[WARNING] LDS cache does not exist")
            shutil.copyfile(generatedReferenceCache, destinationReferenceCache)
            return True

        elif not self._cmp_files(destinationReferenceCache, generatedReferenceCache):
                if self.print_warnings:
                    print(f"[WARNING] LDS cache does not match with reference")
                return False

        if self.print_warnings:
                print(f"[NOTE] LDS cache matches with reference")
        return True


    
    def __find_json_in_console_output(self, console_output):
        match = self.cout_json_regex.search(console_output)
        if match:
            # parse json into a dictionary
            jsonstring = match.group()
            return json.loads(jsonstring)
        elif self.print_warnings:
            print(f"[WARNING] No json output found in pathtracer cout")
        return None


    def __append_before_extension(self, filepath, text):
        filename, extension = os.path.splitext(filepath)
        return filename+text+extension


    def __ssim_test(self, render, reference, difference = None):
        if difference is None:
            difference = "null:" # image magick syntax to not save reference
        else:
            difference = f'"{difference}"'
        command_params = f' compare -metric SSIM "{render}" "{reference}" {difference}'
        command = self.image_magick_exe + command_params
        magickDiffValProcess = subprocess.run(command, capture_output=True)
        similiarity = float(magickDiffValProcess.stderr.decode().strip())
        return 1.0-similiarity
    
    
    def __image_pixel_count(self, image):
        params = f' convert "{image}" -format "%[fx:w*h]" info:'
        executor =  self.image_magick_exe + params
        pixel_count = float(subprocess.run(executor, capture_output=True).stdout.decode().strip())
        return pixel_count


    def __histogram_test(self, render, reference, epsilon, error_threshold_value, allowed_error_pixel_count, error_threshold_type):
        params = f" {render} {reference} -define histogram:unique-colors=true -fx \"(min(u,v)>{epsilon})?((abs(u-v)/min(u,v))>{error_threshold_value}):(max(u,v)>{epsilon})\" -format %c histogram:info:" 
        executor = self.image_magick_exe + params
        error_counter_process = subprocess.run(executor, capture_output=True)

        # first histogram line is the amount of black pixels - the correct ones
        # second (and last) line is amount of white - pixels whose rel err is above NBL_ERROR_THRESHOLD
        histogram_output_lines = error_counter_process.stdout.decode().splitlines()
        error_pixel_count = histogram_output_lines[-1].split()[0][:-1] if len(histogram_output_lines) > 1 else "0"
        if error_threshold_type == ErrorThresholdType.ABSOLUTE:
            passing = float(error_pixel_count) <= float(allowed_error_pixel_count)
            details = f"Errors: {error_pixel_count} / {allowed_error_pixel_count}"
        if self.error_threshold_type == ErrorThresholdType.RELATIVE_TO_RESOLUTION:
            pixel_count = self.__image_pixel_count(str(self.working_dir)+"/"+render)
            error_ratio = float(error_pixel_count) / pixel_count
            passing = error_ratio <= float(allowed_error_pixel_count)
            allowed_error_count = int(float(allowed_error_pixel_count) * pixel_count)
            details = f"Errors: {error_pixel_count} ({error_ratio*100.0:.3f}%) / {allowed_error_count} ({float(allowed_error_pixel_count)*100:.3f}%)"
        return passing, details


    def __create_diff_image(self, render, reference, difference):
        params = f' "{reference}" "{render}" -fx "abs(u-v)" -alpha off "{difference}"'
        command = self.image_magick_exe + params
        subprocess.run(command, capture_output=False)


    def __parse_input_line(self, input_line : str):
        OPTION_PREFIX = "--"
        options = {}
        available_options = [('abs',0),('rel',0),('errcount',1),('errpixel',1),('errssim',1),('epsilon',1)]
        while (input_line.startswith(OPTION_PREFIX)):
            input_line = input_line[len(OPTION_PREFIX)::]
            #parse option
            for op in available_options:
                if input_line.startswith(op[0]):
                    input_line = input_line[len(op[0])::].strip()
                    argv = None
                    if op[1] > 0:
                        index = 0
                        for _ in range(op[1]):
                            index = input_line.index(" ", index) + 1
                        argv = input_line.split(" ",)[::op[1]]
                        input_line = input_line[index::].strip()
                    options[op[0]] = argv
                    break
        return input_line.strip("\n\r "), options            

    # run a test for a single line of input for pathtracer
    def _impl_run_single(self, input_args:str) -> dict:
        executable_arg, options = self.__parse_input_line(input_args)

        # get scene options, either custom or default 
        epsilon = options.get('epsilon', [EPSILON])[0]
        error_threshold_value = options.get('errpixel',[self.error_threshold_value])[0]
        allowed_error_pixel_count = options.get('errcount',[self.allowed_error_pixel_count])[0]
        ssim_error_threshold_value = options.get('errssim',[self.ssim_error_threshold_value])[0]
        error_threshold_type = ErrorThresholdType.RELATIVE_TO_RESOLUTION if 'rel' in options else ErrorThresholdType.ABSOLUTE if 'abs' in options else self.error_threshold_type
        if 'abs' in options and 'rel' in options: # idiot check
            if self.print_warnings:
                print(f"[ERROR] Scene option contain both --rel and bot --abs.")

        results_images = []
        result_status = True
        result_color = "green"
        scene_name = None
        raytracer_bash_command = str(self.executable.absolute()) + ' -SCENE=' + executable_arg + ' -PROCESS_SENSORS RenderAllThenTerminate 0'
        console_output = subprocess.run(raytracer_bash_command, capture_output=True).stdout.decode().strip()
        raytracer_generated_files = self.__find_json_in_console_output(console_output)
        if raytracer_generated_files is not None:
            for render_type, filepath in raytracer_generated_files.items():
                render_type = render_type.strip().removeprefix("output_")
                results_image = {
                    "identifier": render_type,
                    "filename": filepath.split("/")[-1].split("\\")[-1]
                }
                if scene_name is None:
                    scene_name = results_image["filename"].split(".")[0].replace('/', '_').replace('\\','_').removeprefix("Render_")
                    ref_subdir = (('/'+scene_name) if PUT_REFERENCES_IN_FOLDERS else '') 
                    diff_subdir = (('/'+scene_name) if PUT_DIFFERENCES_IN_FOLDERS else '') 
                    render_subdir = (('/'+scene_name) if PUT_RENDERS_IN_FOLDERS else '') 
                reference_filepath = Path(str(self.references_repo_dir) + ref_subdir + '/' + filepath)

                reference_store_filepath = Path(str(self.data_references_abs_dir) + ref_subdir + '/' + filepath)
                if not reference_store_filepath.parent.exists():
                    os.makedirs(reference_store_filepath.parent)
                reference_store_filepath_rel = self.data_references_rel_dir + ref_subdir + '/' + filepath

                render_storage_filepath = Path(str(self.data_renders_abs_dir) + render_subdir + '/' + filepath)
                if not render_storage_filepath.parent.exists():
                    os.makedirs(render_storage_filepath.parent)
                render_store_filepath_rel = self.data_renders_rel_dir + render_subdir + '/' + filepath

                difference_filename = self.__append_before_extension(filepath,"_diff")
                difference_filepath = Path(str(self.data_diffs_abs_dir) + diff_subdir + '/' + difference_filename)
                if not difference_filepath.parent.exists():
                    os.makedirs(difference_filepath.parent)
                difference_store_filepath_rel = self.data_diffs_rel_dir + diff_subdir + '/' + difference_filename

                reference_exists = reference_filepath.exists()

                if not reference_exists:
                    if self.print_warnings:
                        print(f"[WARNING] File {filepath} does not have a reference")
                    results_image["status"] = "passed"
                    results_image["status_color"] = "orange"
                    results_image["details"] = "Missing reference" 
                    result_color = "orange"
                    results_image["render"] = str(render_store_filepath_rel) 
                
                    shutil.copy(filepath, reference_store_filepath)
                    shutil.move(filepath, render_storage_filepath)
                else: 
                    status = True
                    if render_type == "denoised":
                        ssim_diff = self.__ssim_test(filepath, reference_filepath)
                        results_image["details"] = "Difference (SSIM): " + f'{ssim_diff:.4f}'
                        status = ssim_error_threshold_value > ssim_diff
                    else:
                        status, details = self.__histogram_test(filepath, reference_filepath, epsilon, error_threshold_value, allowed_error_pixel_count, error_threshold_type)
                        results_image["details"] = details
                    self.__create_diff_image(filepath, reference_filepath, difference_filepath)
                    results_image["render"] = render_store_filepath_rel
                    results_image["reference"] = reference_store_filepath_rel
                    results_image["difference"] = difference_store_filepath_rel
                    if not status:
                        results_image["status"] = "failed"
                        results_image["status_color"] = "red"
                        result_status = False
                        result_color = "red"
                    else:
                        results_image["status"] = "passed"
                        results_image["status_color"] = "green"
                    shutil.copy(reference_filepath, reference_store_filepath)
                    shutil.move(filepath, render_storage_filepath)
                results_images.append(results_image)
        return {
            'status': 'passed' if result_status else 'failed',
            'status_color': result_color,
            'scene_name': scene_name,
            'array': results_images
        }
    

    # add additional information to the json
    def _impl_append_summary(self, summary: dict):
        summary["lds_cache_hash"] = self.__get_lds_hash()
        summary["error_threshold_type"] =  "absolute" if self.error_threshold_type == ErrorThresholdType.ABSOLUTE else "relative"
        summary["error_threshold_value"] = str(self.error_threshold_value)
        summary["allowed_error_pixel_count"] = str(self.allowed_error_pixel_count)
        summary["ssim_error_threshold_value"] = str(self.ssim_error_threshold_value)
        summary["reference_repo_hash"] = self.__get_ref_repo_hash()
    
    


def run_all_tests(args):
    # test public scenes
    CI_PASS_STATUS_1 = RendersTest(test_name="public",
                    profile="public",
                    executable_filepath=args[0],
                    input_filepath=args[1],
                    nabla_repo_root_dir=args[3],
                    image_magick_exe=args[4],
                    references_repo_dir=args[5],
                    error_threshold_type=ErrorThresholdType.RELATIVE_TO_RESOLUTION,
                    allowed_error_pixel_count=0.0001).run()
    
    #test private scenes
    CI_PASS_STATUS_2 = RendersTest(test_name="private",
                    profile="private",
                    executable_filepath=args[0],
                    input_filepath=args[2],
                    nabla_repo_root_dir=args[3],
                    image_magick_exe=args[4],
                    references_repo_dir=args[6],
                    error_threshold_type=ErrorThresholdType.RELATIVE_TO_RESOLUTION,
                    allowed_error_pixel_count=0.0001).run()

    # check if both were successful
    if not (CI_PASS_STATUS_1 and CI_PASS_STATUS_2):
        print('CI failed')
        exit(-2)
    print('CI done')
    exit()


if __name__ == '__main__':
    run_all_tests(sys.argv[1::])
