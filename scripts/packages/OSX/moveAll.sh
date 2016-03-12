#!/bin/sh
mkdir -p $HOME/Irrlicht1.6.0/
chmod a+rw $HOME/Irrlicht1.6.0
find /private/tmp/Irrlicht1.6.0-install -exec chmod a+rw {} \;
cp -rfp /private/tmp/Irrlicht1.6.0-install/ $HOME/Irrlicht1.6.0/
rm -Rf /private/tmp/Irrlicht1.6.0-install
