mkdir ..\..\..\doctemp
mkdir ..\..\..\doctemp\html
copy doxygen.css ..\..\..\doctemp\html
copy logo.png ..\..\..\doctemp\html


..\doxygen.exe doxygen.cfg

pause
