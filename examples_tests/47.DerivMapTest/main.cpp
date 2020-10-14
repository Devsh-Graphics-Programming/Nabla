// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include "ApplicationHandler.hpp"

int main()
{
	ApplicationHandler application;
	
	if (!application.getStatus())
		return 0;

	application.executeColorSpaceTest();
}