import os
import subprocess
import shutil
import filecmp
from pathlib import *

NBL_IMAGEMAGICK_EXE = Path('@_NBL_IMAGEMAGICK_EXE_@')
NBL_PATHTRACER_EXE = Path('@_NBL_PATHTRACER_EXE_@')
NBL_SCENES_INPUT_TXT = Path('@_NBL_SCENES_INPUT_TXT_@')

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

    if _cacheChanged:
        HTML_BODY += '''
        <h2 style="color: red;">FAILED PASS: Low Discrepancy Sequence Cache has been overwritten by a new one!</h2>
        
        '''
    HTML_BODY += '''
    <table>
      <tr>
        <th>Render</th>
        <th>Pass status</th>
        <th colspan="3" scope="colgroup">Reference</th>
        <th colspan="3" scope="colgroup">Albedo</th>
        <th colspan="3" scope="colgroup">Normal</th>
        <th colspan="3" scope="colgroup">Denoised</th>
      </tr>
    '''

    for _htmlRowTuple in _htmlData:
        HTML_ROW_BODY = '''
        <tr>
          <td>''' + _htmlRowTuple[HTML_TUPLE_RENDER_INDEX] '</td>'

        if _htmlRowTuple[HTML_TUPLE_PASS_STATUS_INDEX]:
            HTML_ROW_BODY += '<td style="color: green;">PASSED</td>'
        else
            HTML_ROW_BODY += '<td style="color: red;">FAILED</td>'

        for i in range(4):
            anIndexOfRenderAspect = i + HTML_TUPLE_REFERENCE_INDEX

            aspectRenderData = _htmlRowTuple[anIndexOfRenderAspect]

            HTML_ROW_BODY += '<td scope="col">' + '<a href="https://TODO.com">' + aspectRenderData[HTML_R_A_N_D_D_DIFF] + '</a></td>' +
            '<td scope="col">An error: ' + aspectRenderData[HTML_R_A_N_D_D_ERROR] + '</td>'

            if aspectRenderData[HTML_R_A_N_D_D_PASS]:
                HTML_ROW_BODY += '<td scope="col" style="color: green;">PASSED</td>'
            else
                HTML_ROW_BODY += '<td scope="col" style="color: red;">FAILED</td>'
        HTML_ROW_BODY += '</tr>'

        HTML_BODY += HTML_ROW_BODY

    HTML_BODY += '''
    </table>
    
    </body>
    </html>
    '''

    htmlFile = open(NBL_CI_WORKING_DIR + '/index.html', "w")
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

        for line in inputLines:
            if list(line)[0] != ';':

                htmlRowTuple = ('', True, ['', '', True], ['', '', True], ['', '', True], ['', '', True])

                renderPath = line.strip().replace('"', '').split()[0]
                htmlRowTuple[HTML_TUPLE_RENDER_INDEX] = renderName = os.path.splitext(str(Path(renderPath).name))[0]
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
                    cacheChanged = True
                    CI_PASS_STATUS = False

                anIndex = HTML_TUPLE_REFERENCE_INDEX
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
                    DIFF_PASS = float(magicDecodeValue) == 0.0
                    if not DIFF_PASS:
                        CI_PASS_STATUS = False
                        htmlRowTuple[HTML_TUPLE_PASS_STATUS_INDEX] = False

                    htmlRowTuple[anIndex][HTML_R_A_N_D_D_DIFF] = renderName + diffTerminator + '_diff.png'
                    htmlRowTuple[anIndex][HTML_R_A_N_D_D_ERROR] = str(magicDecodeValue)
                    htmlRowTuple[anIndex][HTML_R_A_N_D_D_PASS] = DIFF_PASS
                    ++anIndex
                htmlData.append(htmlRowTuple)

        generateHTMLStatus(htmlData, cacheChanged)
    else:
        print('Path tracer executable or scenes input does not exist!')
        exit(-1)

if not CI_PASS_STATUS:
    exit(-1)