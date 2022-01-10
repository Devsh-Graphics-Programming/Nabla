@echo off

set /p imgSz="Image size: "
set /p borderSz="Border size: "
set /a extractedImgSz=imgSz-2*borderSz

set /p img="Image to extract from: "

set /p right="Right path: "
set /p left="Left path: "
set /p top="Top path: "
set /p bottom="Bottom path: "
set /p front="Front path: "
set /p back="Back path: "

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