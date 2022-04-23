import os
import subprocess
import shutil
import filecmp
from pathlib import *

NBL_IMAGEMAGICK_EXE = Path(@_NBL_IMAGEMAGICK_EXE_@)
NBL_PATHTRACER_EXE = Path(@_NBL_PATHTRACER_EXE_@)
NBL_SCENES_INPUT_TXT = Path(@_NBL_SCENES_INPUT_TXT_@)

NBL_CI_WORKING_DIR = Path(str(NBL_PATHTRACER_EXE.parent.absolute()) + '/ci_working_dir')
NBL_CI_REFERENCES_DIR = Path(str(NBL_CI_WORKING_DIR.absolute()) + '/references')
NBL_CI_LDS_CACHE_FILENAME = 'LowDiscrepancySequenceCache.bin'
CI_PASS_STATUS = True

if __name__ == '__main__':
    if NBL_PATHTRACER_EXE.is_file() and NBL_SCENES_INPUT_TXT.is_file():
        with open(NBL_SCENES_INPUT_TXT.absolute()) as aFile:
            inputLines = aFile.readlines()

        os.chdir(NBL_PATHTRACER_EXE.parent.absolute())

        if not NBL_CI_REFERENCES_DIR.is_dir():
            os.makedirs(str(NBL_CI_REFERENCES_DIR.absolute()))

        for line in inputLines:
            if list(line)[0] != ';':
                renderPath = line.strip().replace('"', '').split()[0]
                renderName = os.path.splitext(str(Path(renderPath).name))[0]
                undenoisedTargetName = 'Render_' + renderName + '_scene'

                NBL_DUMMY_CACHE_CASE = not bool(Path(str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + NBL_CI_LDS_CACHE_FILENAME).is_file())
                NBL_DUMMY_RENDER_CASE = not bool(Path(str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + renderName + '/' + undenoisedTargetName + '.exr').is_file())

                # dummy case executes when there is no reference render
                # or no low discrepancy sequence cache in ci working directory
                
                generatedReferenceCache = str(NBL_PATHTRACER_EXE.parent.absolute()) + '/' + NBL_CI_LDS_CACHE_FILENAME
                destinationReferenceCache = str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + NBL_CI_LDS_CACHE_FILENAME
                generatedUndenoisedTargetName = str(NBL_PATHTRACER_EXE.parent.absolute()) + '/' + undenoisedTargetName
                destinationReferenceUndenoisedTargetName = str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + renderName + '/' + undenoisedTargetName

                if NBL_DUMMY_RENDER_CASE:
                    sceneDummyRender = '"' + renderPath + ' ../ci/dummy_4096spp_128depth.xml' + '"'
                    executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + sceneDummyRender + ' -TERMINATE'
                    subprocess.run(executor, capture_output=True)

                    if not Path(destinationReferenceUndenoisedTargetName).parent.is_dir():
                        os.makedirs(str(Path(destinationReferenceUndenoisedTargetName).parent.absolute()))

                    shutil.copyfile(generatedUndenoisedTargetName + '.exr', destinationReferenceUndenoisedTargetName + '.exr')
                    shutil.copyfile(generatedUndenoisedTargetName + '_albedo.exr', destinationReferenceUndenoisedTargetName + '_albedo.exr')
                    shutil.copyfile(generatedUndenoisedTargetName + '_normal.exr', destinationReferenceUndenoisedTargetName + '_normal.exr')
                    shutil.copyfile(generatedUndenoisedTargetName + '_denoised.exr',destinationReferenceUndenoisedTargetName + '_denoised.exr')

                if NBL_DUMMY_CACHE_CASE:
                    shutil.copyfile(generatedReferenceCache, destinationReferenceCache)

                scene = line.strip()
                executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + scene + ' -TERMINATE'
                subprocess.run(executor, capture_output=True)

                # fail CI if the reference cache is different that current generated cache
                if not filecmp.cmp(destinationReferenceCache, generatedReferenceCache):
                    CI_PASS_STATUS = False

                outputDiffTerminators = ['', '_albedo', '_normal', '_denoised']
                for diffTerminator in outputDiffTerminators:
                    diffImageFilepath = str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + renderName + '/' + renderName + diffTerminator + '_diff.png'
                    diffValueFilepath = str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + renderName + '/' + renderName + diffTerminator + '_diff.txt'
                    executor = str(NBL_IMAGEMAGICK_EXE.absolute()) + ' compare -metric AE ' + destinationReferenceUndenoisedTargetName + diffTerminator + '.exr ' + generatedUndenoisedTargetName + diffTerminator + '.exr -compose src ' + diffImageFilepath
                    magicProcess = subprocess.run(executor, capture_output=True)
                    magicDecodeValue = magicProcess.stderr.decode()

                    diffValueFile = open(diffValueFilepath, "w")
                    diffValueFile.write('difference error: ' + str(magicDecodeValue))
                    diffValueFile.close()

                    # threshold for an error, for now we fail CI on any difference
                    if float(magicDecodeValue) != 0.0:
                        CI_PASS_STATUS = False
    else:
        print('Path tracer executable or scenes input does not exist!')
        exit(-1)

if not CI_PASS_STATUS:
    exit(-1)