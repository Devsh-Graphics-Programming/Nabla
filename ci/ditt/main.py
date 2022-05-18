import os
import subprocess
import shutil
import filecmp
from pathlib import *

NBL_IMAGEMAGICK_EXE = Path('@_NBL_IMAGEMAGICK_EXE_@')
NBL_PATHTRACER_EXE = Path('@_NBL_PATHTRACER_EXE_@')
NBL_SCENES_INPUT_TXT = Path('@_NBL_SCENES_INPUT_TXT_@')

NBL_ERROR_THRESHOLD = "0.05" #relative error between reference and generated images, value between 1.0 and 0.0
NBL_ERROR_TOLERANCE_COUNT = 64          

NBL_CI_WORKING_DIR = Path(str(NBL_PATHTRACER_EXE.parent.absolute()) + '/ci_working_dir')
NBL_CI_REFERENCES_DIR = Path(str(NBL_CI_WORKING_DIR.absolute()) + '/references')
NBL_CI_LDS_CACHE_FILENAME = 'LowDiscrepancySequenceCache.bin'
CI_PASS_STATUS = True

HTML_TUPLE_RENDER_INDEX = 0
HTML_TUPLE_PASS_STATUS_INDEX = 1
HTML_TUPLE_REFERENCE_INDEX = 2
HTML_TUPLE_ALBEDO_INDEX = 3
HTML_TUPLE_NORMAL_INDEX = 4
HTML_TUPLE_DENOISED_INDEX = 5

HTML_R_A_N_D_D_DIFF = 0
HTML_R_A_N_D_D_ERROR = 1
HTML_R_A_N_D_D_PASS = 2

def generateHTMLStatus(_htmlData, _cacheChanged):
    HTML_BODY = '''
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
    </head>
    <body>
    
    <h2>Ditt Render Scenes job status</h2>
    
    '''
    HTML_BODY += f'''
    <p>Relative error threshold is set to <strong>{float(NBL_ERROR_THRESHOLD)*100.0}%</strong></p>
    '''
    if _cacheChanged:
        HTML_BODY += '''
        <h2 style="color: red;">FAILED PASS: Low Discrepancy Sequence Cache has been overwritten by a new one!</h2>
        
        '''
    HTML_BODY += '''
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

    for _htmlRowTuple in _htmlData:
        HTML_ROW_BODY = '''
        <tr>
          <td>''' + _htmlRowTuple[HTML_TUPLE_RENDER_INDEX] + '</td>'

        if _htmlRowTuple[HTML_TUPLE_PASS_STATUS_INDEX]:
            HTML_ROW_BODY += '<td style="color: green;">PASSED</td>'
        else:
            HTML_ROW_BODY += '<td style="color: red;">FAILED</td>'

        for i in range(4):
            anIndexOfRenderAspect = i + HTML_TUPLE_REFERENCE_INDEX

            aspectRenderData = _htmlRowTuple[anIndexOfRenderAspect]
            HTML_HYPERLINK_DIFF = 'https://artifactory.devsh.eu/Ditt/ci/data/references/' + _htmlRowTuple[HTML_TUPLE_RENDER_INDEX] + '/' + aspectRenderData[HTML_R_A_N_D_D_DIFF]
            HTML_ROW_BODY += (  '<td scope="col">' + '<a href="' + HTML_HYPERLINK_DIFF + '">' 
                + aspectRenderData[HTML_R_A_N_D_D_DIFF] + '</a></td>' 
                '<td scope="col">Errors: ' + aspectRenderData[HTML_R_A_N_D_D_ERROR] + '</td>')
            if aspectRenderData[HTML_R_A_N_D_D_PASS]:
                HTML_ROW_BODY += '<td scope="col" style="color: green;">PASSED</td>'
            else:
                HTML_ROW_BODY += '<td scope="col" style="color: red;">FAILED</td>'
        HTML_ROW_BODY += '</tr>'

        HTML_BODY += HTML_ROW_BODY

    HTML_BODY += '''
    </table>
    </body>
    </html>
    '''

    htmlFile = open(str(NBL_CI_WORKING_DIR.absolute()) + '/index.html', "w")
    htmlFile.write(HTML_BODY)
    htmlFile.close()


if __name__ == '__main__':
    if NBL_PATHTRACER_EXE.is_file() and NBL_SCENES_INPUT_TXT.is_file():
        with open(NBL_SCENES_INPUT_TXT.absolute()) as aFile:
            inputLines = aFile.readlines()

        os.chdir(NBL_PATHTRACER_EXE.parent.absolute())

        if not NBL_CI_REFERENCES_DIR.is_dir():
            os.makedirs(str(NBL_CI_REFERENCES_DIR.absolute()))

        htmlData = []
        cacheChanged = False

        NBL_DUMMY_CACHE_CASE = not bool(Path(str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + NBL_CI_LDS_CACHE_FILENAME).is_file())
        generatedReferenceCache = str(NBL_PATHTRACER_EXE.parent.absolute()) + '/' + NBL_CI_LDS_CACHE_FILENAME
        destinationReferenceCache = str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + NBL_CI_LDS_CACHE_FILENAME

        sceneDummyRender = '"../ci/dummy_4096spp_128depth.xml"'
        executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + sceneDummyRender + ' -TERMINATE'
        subprocess.run(executor, capture_output=True)
        
        # if we start the path tracer first time
        if NBL_DUMMY_CACHE_CASE:
            shutil.copyfile(generatedReferenceCache, destinationReferenceCache)
        elif not filecmp.cmp(destinationReferenceCache, generatedReferenceCache):
            # fail CI if the reference cache is different that current generated cache
            cacheChanged = True
            CI_PASS_STATUS = False

        for line in inputLines:
            if list(line)[0] != ';':

                htmlRowTuple = ['', True, ['', '', True], ['', '', True], ['', '', True], ['', '', True]]
                renderPath = line.strip().replace('"', '').split()[0]
                renderName = os.path.splitext(str(Path(renderPath).name))[0]
                undenoisedTargetName = 'Render_' + renderName + '_scene'
                NBL_DUMMY_RENDER_CASE = not bool(Path(str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + renderName + '/' + undenoisedTargetName + '.exr').is_file())

                # dummy case executes when there is no reference render
                # or no low discrepancy sequence cache in ci working directory
                generatedUndenoisedTargetName = str(NBL_PATHTRACER_EXE.parent.absolute()) + '/' + undenoisedTargetName
                destinationReferenceUndenoisedTargetName = str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + renderName + '/' + undenoisedTargetName

                # if we render first time a scene then we need to have a reference of this scene for following ci checks
                if NBL_DUMMY_RENDER_CASE:
                    sceneDummyRender = line.strip()
                    executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + sceneDummyRender + ' -TERMINATE'
                    subprocess.run(executor, capture_output=True)
                    if not Path(destinationReferenceUndenoisedTargetName).parent.is_dir():
                        os.makedirs(str(Path(destinationReferenceUndenoisedTargetName).parent.absolute()))
                    shutil.copyfile(generatedUndenoisedTargetName + '.exr', destinationReferenceUndenoisedTargetName + '.exr')
                    shutil.copyfile(generatedUndenoisedTargetName + '_albedo.exr', destinationReferenceUndenoisedTargetName + '_albedo.exr')
                    shutil.copyfile(generatedUndenoisedTargetName + '_normal.exr', destinationReferenceUndenoisedTargetName + '_normal.exr')
                    shutil.copyfile(generatedUndenoisedTargetName + '_denoised.exr',destinationReferenceUndenoisedTargetName + '_denoised.exr')

                scene = line.strip()
                htmlRowTuple[HTML_TUPLE_RENDER_INDEX] = renderName
                executor = str(NBL_PATHTRACER_EXE.absolute()) + ' -SCENE=' + scene + ' -TERMINATE'
                subprocess.run(executor, capture_output=True)

                if not filecmp.cmp(destinationReferenceCache, generatedReferenceCache):
                    # fail CI if the reference cache is different that current generated cache
                    cacheChanged = True
                    CI_PASS_STATUS = False


                anIndex = HTML_TUPLE_REFERENCE_INDEX
                outputDiffTerminators = ['', '_albedo', '_normal', '_denoised']
                for diffTerminator in outputDiffTerminators:
                    imageDiffFilePath = str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + renderName + '/' + renderName + diffTerminator +"_diff.exr"
                    imageRefFilepath = destinationReferenceUndenoisedTargetName + diffTerminator + '.exr'
                    imageGenFilepath = generatedUndenoisedTargetName + diffTerminator + '.exr'
                    
                    #create difference image for debugging
                    diffImageCommandParams = f' "{imageRefFilepath}" "{imageGenFilepath}" -fx "abs(u-v)" "{imageDiffFilePath}"'
                    executor = str(NBL_IMAGEMAGICK_EXE.absolute()) + diffImageCommandParams
                    magickDiffImgProcess = subprocess.run(executor, capture_output=False)

                    #calculate the amount of pixels whose relative errors are above NBL_ERROR_THRESHOLD
                    #logic operators in image magick return 1.0 if true, 0.0 if false 
                    #image magick convert -compose divide does not work with HDRI, this requiring use of -fx 
                    diffValueCommandParams = f" {imageRefFilepath} {imageGenFilepath}  -define histogram:unique-colors=true -fx \"(abs(u-v)/((0!=min(u,v))?abs(min(u,v)):1.0))>={NBL_ERROR_THRESHOLD}\" -format %c histogram:info:" 
                    executor = str(NBL_IMAGEMAGICK_EXE.absolute()) + diffValueCommandParams
                    magickDiffValProcess = subprocess.run(executor, capture_output=True)
                   
                    #first histogram line is the amount of black pixels - the correct ones
                    #second (and last) line is amount of white - pixels whose rel err is above NBL_ERROR_THRESHOLD
                    histogramOutputLines = magickDiffValProcess.stdout.decode().splitlines()
                    errorPixelCount = histogramOutputLines[-1].split()[0][:-1] if len(histogramOutputLines) > 1 else "0"

                    diffValueFilepath = str(NBL_CI_REFERENCES_DIR.absolute()) + '/' + renderName + '/' + renderName + diffTerminator + '_diff.txt'
                    diffValueFile = open(diffValueFilepath, "w")
                    diffValueFile.write('difference error: ' + str(errorPixelCount))
                    diffValueFile.close()

                    # threshold for an error, for now we fail CI when the difference is greater then NBL_ERROR_TOLERANCE_COUNT
                    DIFF_PASS = float(errorPixelCount) <= NBL_ERROR_TOLERANCE_COUNT
                    if not DIFF_PASS:
                        CI_PASS_STATUS = False
                        htmlRowTuple[HTML_TUPLE_PASS_STATUS_INDEX] = False

                    htmlRowTuple[anIndex][HTML_R_A_N_D_D_DIFF] = renderName + diffTerminator + '_diff.exr'
                    htmlRowTuple[anIndex][HTML_R_A_N_D_D_ERROR] = str(errorPixelCount)
                    htmlRowTuple[anIndex][HTML_R_A_N_D_D_PASS] = DIFF_PASS
                    anIndex += 1
                htmlData.append(htmlRowTuple)

        generateHTMLStatus(htmlData, cacheChanged)
    else:
        print('Path tracer executable or scenes input does not exist!')
        exit(-1)

if not CI_PASS_STATUS:
    exit(-2)