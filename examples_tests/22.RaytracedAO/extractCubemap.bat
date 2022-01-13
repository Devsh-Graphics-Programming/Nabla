@echo off

set /p borderSz="Border size: "

set /p img="Image to extract from: "

set /p right="Right path: "
set /p left="Left path: "
set /p top="Top path: "
set /p bottom="Bottom path: "
set /p front="Front path: "
set /p back="Back path: "

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

PAUSE