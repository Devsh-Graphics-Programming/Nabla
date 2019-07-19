#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"

using namespace irr;
using namespace core;

int main()
{
	//pfd::settings::verbose(true);

	pfd::message("Choose file to load", "Choose mitsuba XML file to load. \nIf you cancel or choosen file fails to load bathroom will be loaded.", pfd::choice::ok);
	pfd::open_file file("Choose XML file", "", { "XML files (.xml)", "*.xml" });

	std::string filePath = file.result().empty() ? "C:\\IrrlichtBAW\\/IrrlichtBAW\\examples_tests\\media\\mitsuba\\bathroom\\scene.xml" : file.result()[0];
	
	std::cout << filePath;
	return 0;
}
