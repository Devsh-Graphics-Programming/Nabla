mkdir ..\..\..\doctemp
mkdir ..\..\..\doctemp\html
copy logo.png ..\..\..\doctemp\html

..\doxygen.exe doxygen.cfg

pause

