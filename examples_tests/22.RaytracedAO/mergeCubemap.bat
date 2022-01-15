@echo off

set right=%1
set left=%2
set top=%3
set bottom=%4
set front=%5
set back=%6
set output=%~dpn7

REM examplary usage: 
REM mergeCubemap.bat right.png left.png top.png bottom.png front.png back.png outputImageName

REM set image size
for /f "tokens=*" %%s in ('magick identify -format "%%w" %right%') do set sz=%%s

REM set image fromat
for /f "tokens=*" %%s in ('magick identify -format "%%m" %right%') do set format=%%s

set /a szx2=2*sz
set /a outputWidth=3*sz
set /a outputHeight=2*sz

magick convert -size %outputwidth%x%outputHeight% canvas:none ^
-draw "image over  0,0 0,0 '%right%'" ^
-draw "image over  %sz%,0 0,0 '%left%'" ^
-draw "image over  %szx2%,0 0,0 '%top%'" ^
-draw "image over  0,%sz% 0,0 '%bottom%'" ^
-draw "image over  %sz%,%sz% 0,0 '%front%'" ^
-draw "image over  %szx2%,%sz% 0,0 '%back%'" ^
%output%.%format%