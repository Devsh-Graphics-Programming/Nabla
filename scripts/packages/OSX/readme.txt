This is the source package for the Irrlicht framework installer.

Build steps are as follows-

1) Remove the framework so you don't accidentally package more than one version, if you use the GUI to do this empty the trash so XCode doesn't build to the trash! It's best to run this command from the console:

  rm -Rf /Library/Frameworks/Irrlicht.framework

2) Build the library and the binaries.

 * Navigate to source/Irrlicht/MacOSX and open the XCode project

 * Choose release mode and build Irrlicht.framework

 ** For the moment it's not worth installing the sample binaries, they can't be launched due to console input

3) Build the documentation

 * Open the console and navigate to scripts/doc/irrlicht
 
 * Make sure you have doxygen installed. If you have Aptitude for OSX then type:

  sudo apt-get install doxygen

 * Now run the makedocumentation bash script:
  ./makedocumentation.sh

4) Now double click the package file and build it.
