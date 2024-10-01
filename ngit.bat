for /f "tokens=2,* delims= " %%a in ("%*") do set ALL_BUT_FIRST=%%b
wsl -e git -C "/mnt/d/Nabla/" %ALL_BUT_FIRST%