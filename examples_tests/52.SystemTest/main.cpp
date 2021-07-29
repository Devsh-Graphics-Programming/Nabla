// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>
#include <nbl/ui/CWindowManagerWin32.h>
#include <nbl/system/ISystem.h>
#include "nbl/system/CStdoutLogger.h"
#include "nbl/system/CFileLogger.h"
#include "nbl/system/CColoredStdoutLoggerWin32.h"

using namespace nbl;
using namespace core;
using namespace ui;
using namespace system;
using namespace asset;

class DemoEventCallback : public IWindow::IEventCallback
{
public:
	DemoEventCallback(system::logger_opt_smart_ptr&& logger) : m_logger(std::move(logger)) {}
private:
	void onWindowShown_impl() override 
	{
		m_logger.logI("Window Shown");
	}
	void onWindowHidden_impl() override 
	{
		m_logger.logI("Window hidden");
	}
	void onWindowMoved_impl(int32_t x, int32_t y) override
	{
		m_logger.logW("Window window moved to { %d, %d }", x, y);
	}
	void onWindowResized_impl(uint32_t w, uint32_t h) override
	{
		m_logger.log("Window resized to { %u, %u }", system::ILogger::ELL_DEBUG, w, h);
	}
	void onWindowMinimized_impl() override
	{
		m_logger.logE("Window minimized");
	}
	void onWindowMaximized_impl() override
	{
		m_logger.log("Window maximized", system::ILogger::ELL_PERFORMANCE);
	}
	void onGainedMouseFocus_impl() override
	{
		m_logger.log("Window gained mouse focus", system::ILogger::ELL_INFO);
	}
	void onLostMouseFocus_impl() override
	{
		m_logger.log("Window lost mouse focus", system::ILogger::ELL_INFO);
	}
	void onGainedKeyboardFocus_impl() override
	{
		m_logger.log("Window gained keyboard focus", system::ILogger::ELL_INFO);
	}
	void onLostKeyboardFocus_impl() override
	{
		m_logger.log("Window lost keyboard focus", system::ILogger::ELL_INFO);
	}

	void onMouseConnected_impl(core::smart_refctd_ptr<IMouseEventChannel>&& mch) override
	{
		m_logger.log("A mouse has been connected", system::ILogger::ELL_INFO);
	}
	void onMouseDisconnected_impl(IMouseEventChannel* mch) override
	{
		m_logger.log("A mouse has been disconnected", system::ILogger::ELL_INFO);
	}
	void onKeyboardConnected_impl(core::smart_refctd_ptr<IKeyboardEventChannel>&& kbch) override
	{
		m_logger.log("A keyboard has been connected", system::ILogger::ELL_INFO);
	}
	void onKeyboardDisconnected_impl(IKeyboardEventChannel* mch) override
	{
		m_logger.log("A keyboard has been disconnected", system::ILogger::ELL_INFO);
	}
private:
	system::logger_opt_smart_ptr m_logger;
};

int main()
{
	auto logger = make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>();

	// *** If you don't want logging, uncomment this one line***
	// logger = nullptr;
	// **************************************************************************************
	auto system = ISystem::create();
	auto assetManager = core::make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(system), system::logger_opt_smart_ptr(logger));
	auto winManager = core::make_smart_refctd_ptr<CWindowManagerWin32>();
	
	{
		system::ISystem::future_t<smart_refctd_ptr<system::IFile>> future;
		system->createFile(future, "log.txt", nbl::system::IFile::ECF_READ_WRITE);
		auto file = future.get();
	}

	IWindow::SCreationParams params;
	params.callback = nullptr;
	params.width = 720;
	params.height = 480;
	params.x = 0;
	params.y = 0;
	params.system = core::smart_refctd_ptr(system);
	params.flags = IWindow::ECF_NONE;
	params.windowCaption = "Test Window";

	// *** Select stdout/file logger ***
	params.callback = make_smart_refctd_ptr<DemoEventCallback>(system::logger_opt_smart_ptr(logger));
	//params.callback = make_smart_refctd_ptr<DemoEventCallback>(system::CFileLogger::create(logFileName));
	// *********************************
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
