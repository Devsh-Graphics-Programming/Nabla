// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>
#include <nbl/ui/CWindowManagerWin32.h>
#include <nbl/system/ISystem.h>
using namespace nbl;
using namespace core;
using namespace ui;
using namespace system;
using namespace asset;
int main()
{
	auto system = ISystem::create();
	auto assetManager = core::make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(system));
	auto winManager = core::make_smart_refctd_ptr<CWindowManagerWin32>();
	
	IWindow::SCreationParams params;
	params.callback = nullptr;
	params.width = 720;
	params.height = 480;
	params.x = 0;
	params.y = 0;
	params.system = core::smart_refctd_ptr(system);
	params.flags = IWindow::ECF_NONE;
	params.windowCaption = "Test Window";
	params.callback = make_smart_refctd_ptr<IWindow::IEventCallback>();
	auto window = winManager->createWindow(std::move(params));
	while (true)
	{

	}
}
