@echo off

set /p sz="Image size: "
set /a szx2=2*sz
set /p ext="Image extension: "
set /a outputWidth=3*sz
set /a outputHeight=2*sz

set /p right="Right path: "
set /p left="Left path: "
set /p top="Top path: "
set /p bottom="Bottom path: "
set /p front="Front path: "
set /p back="Back path: "

magick convert -size %outputwidth%x%outputHeight% canvas:skyblue ^
-draw "image over  0,0 0,0 '%right%'" ^
-draw "image over  %sz%,0 0,0 '%left%'" ^
-draw "image over  %szx2%,0 0,0 '%top%'" ^
-draw "image over  0,%sz% 0,0 '%bottom%'" ^
-draw "image over  %sz%,%sz% 0,0 '%front%'" ^
-draw "image over  %szx2%,%sz% 0,0 '%back%'" ^
output.%ext%

PAUSE