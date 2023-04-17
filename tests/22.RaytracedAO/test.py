from CITest import *


CLOSE_TO_ZERO = "0.00001"         
CI_PASS_STATUS = True


class RendersTest(CITest):
    # self.executable 
    # self.input 
    # self.print_warnings
    # self.nabla_repo_root_dir
    
    # overwrite a function called in ctor 
    # make sure all needed dirs are created
    def _validate_filepaths(self):
        if not CITest._validate_filepaths():
            return False
        if not self.references_dir.is_dir():
            os.makedirs(self.references_dir)

        if not self.storage_dir.is_dir():
            os.makedirs(self.storage_dir)
                
        if not self.diff_images_dir.is_dir():
            os.makedirs(self.diff_images_dir)


    def __call_executor(self):
        executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + sceneDummyRender + ' -PROCESS_SENSORS RenderAllThenTerminate 0'
        subprocess.run(executor, capture_output=True)
    
    
    def _impl_run_dummy_case(self):
        NBL_DUMMY_CACHE_CASE = not bool(Path(str(inputParams.references_dir) + '/' + NBL_REF_LDS_CACHE_FILENAME).is_file())
        generatedReferenceCache = str(NBL_PATHTRACER_EXE.parent.absolute()) + '/' + NBL_REF_LDS_CACHE_FILENAME
        destinationReferenceCache = str(inputParams.references_dir) + '/' + NBL_REF_LDS_CACHE_FILENAME
        sceneDummyRender = '"../ci/dummy_4096spp_128depth.xml"'
        executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + sceneDummyRender + ' -PROCESS_SENSORS RenderAllThenTerminate 0'
        subprocess.run(executor, capture_output=True)

        if NBL_DUMMY_CACHE_CASE:
                shutil.copyfile(generatedReferenceCache, destinationReferenceCache)
        elif not cmp_files(inputParams,destinationReferenceCache, generatedReferenceCache):
                cacheChanged = True
                ci_pass_status = False


    # must override 
    def _impl_run_single(self, input_args) -> dict:
        pass


NBL_IMAGEMAGICK_EXE = Path('@_NBL_IMAGEMAGICK_EXE_@')
NBL_PATHTRACER_EXE = Path('@_NBL_PATHTRACER_EXE_@')
NBL_REFDATA_PATH = '@NBL_ROOT_PATH@'+'/ci/22.RaytracedAO' # TODO: change `ci` to `tests/reference_data`
NBL_REF_LDS_CACHE_FILENAME = 'LowDiscrepancySequenceCache.bin'
NBL_ERROR_THRESHOLD = "0.05" #relative error between reference and generated images, value between 1.0 and 0.0
NBL_ERROR_TOLERANCE_COUNT = 96   #TODO: make this relative to image resolution
 

#path to dir containing Nabla source
ROOT_PATH =  '@NBL_ROOT_PATH@'
input_file='@NBL_ROOT_PATH@'+'/examples_tests/media/mitsuba/public_test_scenes.txt',
summary_html_filepath=f'{NBL_REFDATA_PATH}/renders/public/index.html', 
ref_url='https://github.com/Devsh-Graphics-Programming/Nabla-Ci/tree/'+ get_git_revision_hash() + '/22.RaytracedAO/references/public',
diff_imgs_url = 'https://artifactory.devsh.eu/Ditt/ci/data/renders/public/difference-images',
result_imgs_url = 'https://artifactory.devsh.eu/Ditt/ci/data/renders/public',
references_dir=f'{NBL_REFDATA_PATH}/references/public',
diff_images_dir=f'{NBL_REFDATA_PATH}/renders/public/difference-images',
storage_dir= f'{NBL_REFDATA_PATH}/renders/public'


input_file='@NBL_ROOT_PATH@'+'/examples_tests/media/Ditt-Reference-Scenes/private_test_scenes.txt',
summary_html_filepath=f'{NBL_REFDATA_PATH}/renders/private/index.html', 
ref_url='https://github.com/Devsh-Graphics-Programming/Ditt-Reference-Renders/tree/' + get_submodule_revision_hash(),
diff_imgs_url = 'https://artifactory.devsh.eu/Ditt/ci/data/renders/private/difference-images',
result_imgs_url = 'https://artifactory.devsh.eu/Ditt/ci/data/renders/private',
references_dir=f'{NBL_REFDATA_PATH}/references/private',
diff_images_dir=f'{NBL_REFDATA_PATH}/renders/private/difference-images',
storage_dir= f'{NBL_REFDATA_PATH}/renders/private'
]



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
            executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + sceneDummyRender + ' -PROCESS_SENSORS RenderAllThenTerminate 0'
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

                        destinationReferenceUndenoisedTargetName = str(inputParams.references_dir) + '/' + renderName + '/' + undenoisedTargetName
                    
                        NBL_DUMMY_RENDER_CASE = not bool(Path(destinationReferenceUndenoisedTargetName + '.exr').is_file())

                        executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + scene + ' -PROCESS_SENSORS RenderAllThenTerminate 0'
                        subprocess.run(executor, capture_output=True)

                        outputDiffTerminators = ['', '_albedo', '_normal', '_denoised']
                        HTML_CELLS = []
                        PASSED_ALL = True
                        storageFilepath = str(inputParams.storage_dir) + '/' + undenoisedTargetName

                        generatedUndenoisedTargetName = str(NBL_PATHTRACER_EXE.parent.absolute()) + '/' + undenoisedTargetName
                        if not Path(generatedUndenoisedTargetName+".exr").is_file():
                            generatedUndenoisedTargetName = str(NBL_PATHTRACER_EXE.parent.absolute()) + '/' + renderName
                        referenceDir = str(inputParams.storage_dir) + '/references/' + renderName + '/'
                        if not Path(referenceDir).is_dir():
                            os.makedirs(referenceDir)

                        for diffTerminator in outputDiffTerminators:
                            try:
                                imageDiffFilePath = str(inputParams.diff_images_dir) + '/' + renderName + diffTerminator + "_diff.exr"
                                imageRefFilepath = destinationReferenceUndenoisedTargetName + diffTerminator + '.exr'
                                imageGenFilepath = generatedUndenoisedTargetName + diffTerminator + '.exr'
                                refStorageFilepathstr = referenceDir + undenoisedTargetName + diffTerminator + '.exr' 
                                
                                if not Path(imageGenFilepath).is_file():
                                    HTML_CELL = f'''
                                    <td scope="col">
                                        <td scope="col">Failed to produce render</td>
                                        <td style="color: red;">CRITICAL</td>
                                    </td>
                                    '''
                                    HTML_CELLS.append(HTML_CELL)
                                    ci_pass_status = False
                                    PASSED_ALL = False
                                    continue
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
                                    shutil.copy(generatedUndenoisedTargetName + diffTerminator +'.exr', storageFilepath + diffTerminator + '.exr')
                                    #fix to CORS in preview
                                    shutil.move(generatedUndenoisedTargetName + diffTerminator +'.exr', refStorageFilepathstr)
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
                                
                                #fix to CORS in preview
                                shutil.copy(imageRefFilepath, refStorageFilepathstr)
                                
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

                            HTML_CELLS.append(HTML_CELL)
                        
                        #write to file in append mode 
                        HTML = f'''
                        <tr>
                        <td>{renderName}</td>
                        {'<td style="color: green;">PASSED</td>' if PASSED_ALL else '<td style="color: red;">FAILED</td>'}
                        ''' + ' '.join(HTML_CELLS)  + f'''
                        <td scope="col"><button type="button" class="btn" onclick="compareImageBtnCallback('{renderName}');">Compare</button></td>

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