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

	system::ISystem::future_t<smart_refctd_ptr<system::IFile>> future;
	system->createFile(future, "testFile.txt", nbl::system::IFile::ECF_READ_WRITE);
	auto file = future.get();
	std::string fileData = "Test file data!";

	system::ISystem::future_t<uint32_t> writeFuture;
	system->writeFile(writeFuture, file.get(), fileData.data(), 0, fileData.length());
	assert(writeFuture.get() == fileData.length());

	std::string readStr(fileData.length(), '\0');
	system::ISystem::future_t<uint32_t> readFuture;
	system->readFile(readFuture, file.get(), readStr.data(), 0, readStr.length());
	assert(readFuture.get() == fileData.length());

	while (true)
	{

	}
}
