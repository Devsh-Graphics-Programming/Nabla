@echo off

REM examplary usage: 
REM mergeCubemap.bat 50 mergedImage.png right.png left.png top.png bottom.png front.png back.png

set borderSz=%1

set img=%2

set right=%3
set left=%4
set top=%5
set bottom=%6
set front=%7
set back=%8

REM set extracted image size
for /f "tokens=*" %%s in ('magick identify -format "%%w" %img%') do set sz=%%s
set /a imgSz=sz/3
set /a extractedImgSz=imgSz-2*borderSz

set /a x0 = borderSz
set /a x1 = imgSz+borderSz
set /a x2 = 2*imgSz+borderSz
set /a y0 = borderSz
set /a y1 = imgSz+borderSz

magick convert %img% -crop %extractedImgSz%x%extractedImgSz%+%x0%+%y0% %right%
magick convert %img% -crop %extractedImgSz%x%extractedImgSz%+%x1%+%y0% %left%
magick convert %img% -crop %extractedImgSz%x%extractedImgSz%+%x2%+%y0% %top%
magick convert %img% -crop %extractedImgSz%x%extractedImgSz%+%x0%+%y1% %bottom%
magick convert %img% -crop %extractedImgSz%x%extractedImgSz%+%x1%+%y1% %front%
magick convert %img% -crop %extractedImgSz%x%extractedImgSz%+%x2%+%y1% %back%