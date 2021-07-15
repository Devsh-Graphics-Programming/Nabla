// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>
#include <nbl/ui/CWindowManagerWin32.h>
#include <nbl/system/ISystem.h>
#include <CLogger.h>
using namespace nbl;
using namespace core;
using namespace ui;
using namespace system;
using namespace asset;
using namespace os;

// Don't wanna use Printer::log
#define LOG(...) printf(__VA_ARGS__); printf("\n");
class DemoEventCallback : public IWindow::IEventCallback
{
	void onWindowShown_impl() override 
	{
		LOG("Window Shown");
	}
	void onWindowHidden_impl() override 
	{
		LOG("Window hidden");
	}
	void onWindowMoved_impl(int32_t x, int32_t y) override
	{
		LOG("Window window moved to { %d, %d }", x, y);
	}
	void onWindowResized_impl(uint32_t w, uint32_t h) override
	{
		LOG("Window resized to { %u, %u }", w, h);

	}
	void onWindowMinimized_impl() override
	{
		LOG("Window minimized");
	}
	void onWindowMaximized_impl() override
	{
		LOG("Window maximized");
	}
	void onGainedMouseFocus_impl() override
	{
		LOG("Window gained mouse focus");
	}
	void onLostMouseFocus_impl() override
	{
		LOG("Window lost mouse focus");
	}
	void onGainedKeyboardFocus_impl() override
	{
		LOG("Window gained keyboard focus");
	}
	void onLostKeyboardFocus_impl() override
	{
		LOG("Window lost keyboard focus");
	}

	void onMouseConnected_impl(core::smart_refctd_ptr<IMouseEventChannel>&& mch) override
	{
		LOG("A mouse has been connected");
	}
	void onMouseDisconnected_impl(IMouseEventChannel* mch) override
	{
		LOG("A mouse has been disconnected");
	}
	void onKeyboardConnected_impl(core::smart_refctd_ptr<IKeyboardEventChannel>&& kbch) override
	{
		LOG("A keyboard has been connected");
	}
	void onKeyboardDisconnected_impl(IKeyboardEventChannel* mch) override
	{
		LOG("A keyboard has been disconnected");
	}
};

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
	params.callback = make_smart_refctd_ptr<DemoEventCallback>();
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
