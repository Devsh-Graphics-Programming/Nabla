#define _IRR_STATIC_LIB_
#include "ApplicationHandler.hpp"

int main()
{
	ApplicationHandler application;
	
	if (!application.getStatus())
		return 0;

	application.executeColorSpaceTest();
}