mkdir ..\..\..\doctemp
mkdir ..\..\..\doctemp\html
copy doxygen.css ..\..\..\doctemp\html
copy irrlicht.png ..\..\..\doctemp\html
copy logobig.png ..\..\..\doctemp\html

rem for /F %%i in ('dir ..\..\..\examples\[01]*\main.cpp') DO ..\sed.exe -f tutorials.sed %i >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\01.HelloWorld\main.cpp >tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\02.Quake3Map\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\03.CustomSceneNode\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\04.Movement\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\05.UserInterface\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\06.2DGraphics\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\07.Collision\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\08.SpecialFX\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\09.MeshViewer\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\10.Shaders\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\11.PerPixelLighting\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\12.TerrainRendering\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\13.RenderToTexture\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\14.Win32Window\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\15.LoadIrrFile\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\16.Quake3MapShader\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\17.HelloWorld_Mobile\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\18.SplitScreen\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\19.MouseAndJoystick\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\20.ManagedLights\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\21.Quake3Explorer\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\22.MaterialViewer\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\23.SMeshHandling\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\24.CursorControl\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\25.XmlHandling\main.cpp >>tut.txt
..\sed.exe -f tutorials.sed ..\..\..\examples\26.OcclusionQuery\main.cpp >>tut.txt

..\doxygen.exe doxygen.cfg

pause
