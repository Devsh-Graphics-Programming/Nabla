from operator import eq
import os
import subprocess
import shutil
import filecmp
from datetime import datetime
from pathlib import *

NBL_IMAGEMAGICK_EXE = Path('@_NBL_IMAGEMAGICK_EXE_@')
NBL_PATHTRACER_EXE = Path('@_NBL_PATHTRACER_EXE_@')
NBL_REFDATA_PATH = '@NBL_ROOT_PATH@'+'/ci/22.RaytracedAO' # TODO: change `ci` to `tests/reference_data`
NBL_REF_LDS_CACHE_FILENAME = 'LowDiscrepancySequenceCache.bin'
NBL_ERROR_THRESHOLD = "0.05" #relative error between reference and generated images, value between 1.0 and 0.0
NBL_ERROR_TOLERANCE_COUNT = 96   #TODO: make this relative to image resolution
 
def get_git_revision_hash() -> str:
    return subprocess.check_output(f'git -C "{NBL_REFDATA_PATH}" rev-parse origin/ditt').decode('ascii').strip()

def get_submodule_revision_hash() -> str:
    return subprocess.check_output(f'git -C "{NBL_REFDATA_PATH}/references/private" rev-parse origin/master').decode('ascii').strip()

class Inputs:
    def __init__(self, 
                input_file: Path,
                ref_url: str,
                diff_imgs_url: str,
                result_imgs_url: str,
                summary_html_filepath: Path,
                references_dir: str,
                diff_images_dir: str,
                storage_dir: str) -> None:
        self.input_file_path = Path(input_file).absolute()
        self.ref_url = ref_url
        self.diff_imgs_url = diff_imgs_url
        self.result_imgs_url = result_imgs_url
        self.summary_html_filepath = Path(summary_html_filepath).absolute()
        self.references_dir = Path(references_dir).absolute()
        self.diff_images_dir = Path(diff_images_dir).absolute()
        self.storage_dir = Path(storage_dir).absolute()

NBL_SCENES_INPUTS = [ 
    Inputs(
            input_file='@NBL_ROOT_PATH@'+'/examples_tests/media/mitsuba/public_test_scenes.txt',
            summary_html_filepath=f'{NBL_REFDATA_PATH}/renders/public/index.html', 
            ref_url='https://github.com/Devsh-Graphics-Programming/Nabla-Ci/tree/'+ get_git_revision_hash() + '/22.RaytracedAO/references/public',
            diff_imgs_url = 'https://artifactory.devsh.eu/Ditt/ci/data/renders/public/difference-images',
            result_imgs_url = 'https://artifactory.devsh.eu/Ditt/ci/data/renders/public',
            references_dir=f'{NBL_REFDATA_PATH}/references/public',
            diff_images_dir=f'{NBL_REFDATA_PATH}/renders/public/difference-images',
            storage_dir= f'{NBL_REFDATA_PATH}/renders/public'),

        Inputs(
            input_file='@_NBL_PRIVATE_SCENES_INPUT_TXT_@', 
            summary_html_filepath=f'{NBL_REFDATA_PATH}/renders/private/index.html', 
            ref_url='https://github.com/Devsh-Graphics-Programming/Ditt-Reference-Renders/tree/' + get_submodule_revision_hash(),
            diff_imgs_url = 'https://artifactory.devsh.eu/Ditt/ci/data/renders/private/difference-images',
            result_imgs_url = 'https://artifactory.devsh.eu/Ditt/ci/data/renders/private',
            references_dir=f'{NBL_REFDATA_PATH}/references/private',
            diff_images_dir=f'{NBL_REFDATA_PATH}/renders/private/difference-images',
            storage_dir= f'{NBL_REFDATA_PATH}/renders/private') 
]
CLOSE_TO_ZERO = "0.00001"         
CI_PASS_STATUS = True


def htmlHead(scenes_input: Inputs):
    HTML = '''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    table {
      font-family: arial, sans-serif;
      border-collapse: collapse;
      width: 100%;
    }
    
    td, th {
      border: 1px solid #dddddd;
      text-align: left;
      padding: 8px;
      text-align: center;
    }
    
    tr:nth-child(even) {
      background-color: #dddddd;
    }
    
    table, th, td {
        border: 1px solid black;
    }
    </style>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
    <meta http-equiv="Pragma" content="no-cache" />
    <meta http-equiv="Expires" content="0" />
    </head>
    <body>
    
    <h2>Ditt Render Scenes job status</h2>
    '''
    HTML += f'''
    <p>Relative error threshold is set to <strong>{float(NBL_ERROR_THRESHOLD)*100.0}%</strong></p>
    <p>Created at {datetime.now()} </p>
    <table>
      <tr>
        <th>Render</th>
        <th>Pass status</th>
        <th colspan="3" scope="colgroup">Input</th>
        <th colspan="3" scope="colgroup">Albedo</th>
        <th colspan="3" scope="colgroup">Normal</th>
        <th colspan="3" scope="colgroup">Denoised</th>
      </tr>
    '''
    htmlFile = open(scenes_input.summary_html_filepath, "w+")
    htmlFile.write(HTML)

    return HTML

def htmlFoot(_cacheChanged : bool, scenes_input : Inputs):
    HTML = '</table>'

    if _cacheChanged:
        HTML += '''
        <h2 style="color: red;">FAILED PASS: Low Discrepancy Sequence Cache has been overwritten by a new one!</h2>
        '''
    else:
        executor = f'git hash-object {scenes_input.references_dir}/{NBL_REF_LDS_CACHE_FILENAME}'
        hash = subprocess.run(executor, capture_output=True).stdout.decode().strip()
        HTML += f'''
        <h2 style="color: green;">LDS Cache hash: {hash}</h2>'''
    HTML += '''
    </body>
    </html>
    '''
    htmlFile = open(scenes_input.summary_html_filepath, "a")
    htmlFile.write(HTML)
    htmlFile.close()

def get_render_filename(line : str):
    words = line.replace('"', '').strip().split(" ")
    zip = (os.path.splitext(str(Path(" ".join(words[0:-1])).name))[0] + "_") if len(words) > 1 else "" 
    return zip + os.path.splitext(Path(words[-1]).name)[0]


def cmp_files(inputParams, destinationReferenceCache, generatedReferenceCache, cmpSavedHash=False, cmpByteByByte = False):
    if cmpByteByByte:
        return  filecmp.cmp(destinationReferenceCache, generatedReferenceCache)
    executor1 = f'git hash-object {generatedReferenceCache}'
    executor2 = f'git hash-object {destinationReferenceCache}'
    hgen = subprocess.run(executor1, capture_output=True).stdout.decode().strip()
    href = subprocess.run(executor2, capture_output=True).stdout.decode().strip()
    res = hgen == href
    if cmpSavedHash:
        file = str(inputParams.references_dir)+'/LDSCacheHash.txt'
        if Path(file).is_file():
            with open(file, "r") as f:
                res = res and hgen == f.readline().strip()
    return res


def run_all_tests(inputParamList):
    ci_pass_status = True
    if NBL_PATHTRACER_EXE.is_file():

        os.chdir(NBL_PATHTRACER_EXE.parent.absolute()) 

        for inputParams in inputParamList:
            
            if not inputParams.references_dir.is_dir():
                os.makedirs(inputParams.references_dir)

            if not inputParams.storage_dir.is_dir():
                os.makedirs(inputParams.storage_dir)
                
            if not inputParams.diff_images_dir.is_dir():
                os.makedirs(inputParams.diff_images_dir)

            NBL_DUMMY_CACHE_CASE = not bool(Path(str(inputParams.references_dir) + '/' + NBL_REF_LDS_CACHE_FILENAME).is_file())
            generatedReferenceCache = str(NBL_PATHTRACER_EXE.parent.absolute()) + '/' + NBL_REF_LDS_CACHE_FILENAME
            destinationReferenceCache = str(inputParams.references_dir) + '/' + NBL_REF_LDS_CACHE_FILENAME
            cacheChanged = False

            sceneDummyRender = '"../ci/dummy_4096spp_128depth.xml"'
            executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + sceneDummyRender + ' -TERMINATE'
            subprocess.run(executor, capture_output=True)
                
            # if we start the path tracer first time
            if NBL_DUMMY_CACHE_CASE:
                shutil.copyfile(generatedReferenceCache, destinationReferenceCache)
            # fail CI if the reference cache is different that current generated cache
            elif not cmp_files(inputParams,destinationReferenceCache, generatedReferenceCache):
                cacheChanged = True
                ci_pass_status = False
                continue

            input_filepath = inputParams.input_file_path
            if not input_filepath.is_file():
                print(f'Scenes input {str(input_filepath)} does not exist!')
                continue
        
            with open(input_filepath.absolute()) as aFile:
                inputLines = aFile.readlines()

            htmlHead(inputParams)
            for line in inputLines:
                if list(line)[0] != ';':
                    try:
                        renderName = get_render_filename(line)
                        undenoisedTargetName = 'Render_' + renderName
                       
                        scene = line.strip()

                        generatedUndenoisedTargetName = str(NBL_PATHTRACER_EXE.parent.absolute()) + '/' + undenoisedTargetName
                        destinationReferenceUndenoisedTargetName = str(inputParams.references_dir) + '/' + renderName + '/' + undenoisedTargetName
                    
                        # dummy case executes when there is no reference image
                        NBL_DUMMY_RENDER_CASE = not bool(Path(destinationReferenceUndenoisedTargetName + '.exr').is_file())


                        executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + scene + ' -TERMINATE'
                        subprocess.run(executor, capture_output=True)

                        # fail CI if the reference cache is different that current generated cache
                    
                        outputDiffTerminators = ['', '_albedo', '_normal', '_denoised']
                        HTML_CELLS = []
                        PASSED_ALL = True
                        storageFilepath = str(inputParams.storage_dir) + '/' + undenoisedTargetName

                        for diffTerminator in outputDiffTerminators:
                            try:
                                imageDiffFilePath = str(inputParams.diff_images_dir) + '/' + renderName + diffTerminator + "_diff.exr"
                                imageRefFilepath = destinationReferenceUndenoisedTargetName + diffTerminator + '.exr'
                                imageGenFilepath = generatedUndenoisedTargetName + diffTerminator + '.exr'
                                                        # if we render first time a scene then we need to have a reference of this scene for following ci checks
                                if NBL_DUMMY_RENDER_CASE:

                                    HTML_CELL = f'''
                                    <td scope="col">
                                    <a href="{inputParams.result_imgs_url}/{undenoisedTargetName}{diffTerminator}.exr">(Result)</a>
                                    <td scope="col">No references</td>
                                    <td style="color: orange;">PASSED</td>
                                    </td>
                                    '''
                                    HTML_CELLS.append(HTML_CELL)
                                    shutil.move(generatedUndenoisedTargetName + diffTerminator +'.exr', storageFilepath + diffTerminator + '.exr')
                                    continue

                                if diffTerminator =='_denoised':
                                    diffValueCommandParams = f' compare -metric SSIM "{imageRefFilepath}" "{imageGenFilepath}" "{imageDiffFilePath}"'
                                    executor = str(NBL_IMAGEMAGICK_EXE.absolute()) + diffValueCommandParams
                                    magickDiffValProcess = subprocess.run(executor, capture_output=True)
                                    similiarity = float(magickDiffValProcess.stderr.decode().strip())
                                    DIFF_PASS = 1.0-similiarity <= float(NBL_ERROR_THRESHOLD)
                                    TAB3 = "Similiarity: "+ str(similiarity*100.0) + "%" 
                                else:
                                    #create difference image for debugging
                                    diffImageCommandParams = f' "{imageRefFilepath}" "{imageGenFilepath}" -fx "abs(u-v)" -alpha off "{imageDiffFilePath}"'
                                    executor = str(NBL_IMAGEMAGICK_EXE.absolute()) + diffImageCommandParams
                                    subprocess.run(executor, capture_output=False)

                                    #calculate the amount of pixels whose relative errors are above NBL_ERROR_THRESHOLD
                                    #logic operators in image magick return 1.0 if true, 0.0 if false 
                                    #image magick convert -compose divide does not work with HDRI, this requiring use of -fx 
                                    diffValueCommandParams = f" {imageRefFilepath} {imageGenFilepath}  -define histogram:unique-colors=true -fx \"(min(u,v)>{CLOSE_TO_ZERO})?((abs(u-v)/min(u,v))>{NBL_ERROR_THRESHOLD}):(max(u,v)>{CLOSE_TO_ZERO})\" -format %c histogram:info:" 
                                    executor = str(NBL_IMAGEMAGICK_EXE.absolute()) + diffValueCommandParams
                                    magickDiffValProcess = subprocess.run(executor, capture_output=True)
                                
                                    #first histogram line is the amount of black pixels - the correct ones
                                    #second (and last) line is amount of white - pixels whose rel err is above NBL_ERROR_THRESHOLD
                                    histogramOutputLines = magickDiffValProcess.stdout.decode().splitlines()
                                    errorPixelCount = histogramOutputLines[-1].split()[0][:-1] if len(histogramOutputLines) > 1 else "0"

                                    # threshold for an error, for now we fail CI when the difference is greater then NBL_ERROR_TOLERANCE_COUNT
                                    DIFF_PASS = float(errorPixelCount) <= NBL_ERROR_TOLERANCE_COUNT
                                    TAB3 = "Errors: " + str(errorPixelCount)

                                shutil.move(generatedUndenoisedTargetName + diffTerminator +'.exr', storageFilepath + diffTerminator + '.exr')

                                Diff_Filename= renderName + diffTerminator + "_diff.exr"
                                HTML_CELL = f'''
                                <td scope="col">
                                <a href="{inputParams.diff_imgs_url}/{Diff_Filename}">(Difference)</a><br>
                                <a href="{inputParams.ref_url}/{renderName}/Render_{renderName}{diffTerminator}.exr">(Reference)</a><br>
                                <a href="{inputParams.result_imgs_url}/{undenoisedTargetName}{diffTerminator}.exr">(Result)</a>
                                <td scope="col">{TAB3}</td>
                                {'<td style="color: green;">PASSED</td>' if DIFF_PASS else '<td style="color: red;">FAILED</td>'}
                                </td>
                                '''
                                if not DIFF_PASS:
                                    ci_pass_status = False
                                    PASSED_ALL = False
                                print (f'\t\t{renderName}{diffTerminator}   {"PASSED" if DIFF_PASS else "FAILED"}')

                            except Exception as ex:
                                
                                print(f"Exception occured inside an innermost loop during rendering {renderName}{diffTerminator}: {str(ex)}")
                                raise ex

                            HTML_CELLS.append(HTML_CELL)
                        
                        #write to file in append mode 
                        HTML = f'''
                        <tr>
                        <td>{renderName}</td>
                        {'<td style="color: green;">PASSED</td>' if PASSED_ALL else '<td style="color: red;">FAILED</td>'}
                        ''' + ' '.join(HTML_CELLS)  + '''
                        </tr>
                        '''
                        print (f'Overall {renderName}   {"PASSED" if PASSED_ALL else "FAILED"}')


                        storageFilepath = str(inputParams.storage_dir) + '/' + undenoisedTargetName
                    except Exception as ex:
                        HTML = f'''<tr style="color: red;">CRASHED</tr>'''
                        print(f"Critical exception occured during rendering {line}: {str(ex)}")
                        ci_pass_status = False
                        break
                    htmlFile = open(inputParams.summary_html_filepath, "a")
                    htmlFile.write(HTML)
                    htmlFile.close()


            if not cmp_files(inputParams,destinationReferenceCache, generatedReferenceCache):
                cacheChanged = True
                ci_pass_status = False
            htmlFoot(cacheChanged, inputParams)
    else:
        print('Path tracer executable does not exist!')
        exit(-1)
    return ci_pass_status

if __name__ == '__main__':
    CI_PASS_STATUS=run_all_tests(NBL_SCENES_INPUTS)
    print('CI done')


if not CI_PASS_STATUS:
    exit(-2)
