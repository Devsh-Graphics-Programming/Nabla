@echo off

REM examplary usage: 
REM mergeCubemap.bat 64 64 1024 1024 mergedImage.png right.png left.png top.png bottom.png front.png back.png

set cropOffsetX=%1
set cropOffsetY=%2

set cropWidth=%3
set cropHeight=%4

set img=%5

set right=%6
set left=%7
set top=%8
set bottom=%9

REM you can't do %10 %11 in batch file
shift
set front=%9
shift
set back=%9

REM set extracted image size
for /f "tokens=*" %%s in ('magick identify -format "%%w" %img%') do set sz=%%s
set /a imgSz=sz/3

set /a x0 = cropOffsetX
set /a x1 = imgSz+cropOffsetX
set /a x2 = 2*imgSz+cropOffsetX
set /a y0 = cropOffsetY
set /a y1 = imgSz+cropOffsetY

magick convert %img% -crop %cropWidth%x%cropHeight%+%x0%+%y0% %right%
magick convert %img% -crop %cropWidth%x%cropHeight%+%x1%+%y0% %left%
magick convert %img% -crop %cropWidth%x%cropHeight%+%x2%+%y0% %top%
magick convert %img% -crop %cropWidth%x%cropHeight%+%x0%+%y1% %bottom%
magick convert %img% -crop %cropWidth%x%cropHeight%+%x1%+%y1% %front%
magick convert %img% -crop %cropWidth%x%cropHeight%+%x2%+%y1% %back%