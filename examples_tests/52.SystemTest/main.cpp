// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>
#include <nbl/ui/CWindowManagerWin32.h>
#include <nbl/system/ISystem.h>
#include "../common/CommonAPI.h"
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
		m_logger.log("Window Shown");
	}
	void onWindowHidden_impl() override 
	{
		m_logger.log("Window hidden");
	}
	void onWindowMoved_impl(int32_t x, int32_t y) override
	{
		m_logger.log("Window window moved to { %d, %d }", system::ILogger::ELL_WARNING, x, y);
	}
	void onWindowResized_impl(uint32_t w, uint32_t h) override
	{
		m_logger.log("Window resized to { %u, %u }", system::ILogger::ELL_DEBUG, w, h);
	}
	void onWindowMinimized_impl() override
	{
		m_logger.log("Window minimized", system::ILogger::ELL_ERROR);
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
	auto system = CommonAPI::createSystem();
	auto assetManager = core::make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(system), system::logger_opt_smart_ptr(logger));

	// *** If you don't want logging, uncomment this one line***
	// logger = nullptr;
	// **************************************************************************************

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

	system::future<size_t> writeFuture;
	file->write(writeFuture, fileData.data(), 0, fileData.length());
	assert(writeFuture.get() == fileData.length());

	std::string readStr(fileData.length(), '\0');
	system::future<size_t> readFuture;
	file->read(readFuture, readStr.data(), 0, readStr.length());
	assert(readFuture.get() == fileData.length());

	//PNG loader test
	{
		IAssetLoader::SAssetLoadParams lp;
		auto asset = assetManager->getAsset("../../media/cegui_alfisko/screenshot.png", lp);
		assert(!asset.getContents().empty());
		auto cpuImage = static_cast<ICPUImage*>(asset.getContents().begin()->get());
		core::smart_refctd_ptr<ICPUImageView> imageView;
		ICPUImageView::SCreationParams imgViewParams;
		imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		imgViewParams.format = E_FORMAT::EF_R8G8B8_UINT;
		imgViewParams.image = core::smart_refctd_ptr<ICPUImage>(cpuImage);
		imgViewParams.viewType = ICPUImageView::ET_2D;
		imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
		imageView = ICPUImageView::create(std::move(imgViewParams));

		IAssetWriter::SAssetWriteParams wp(imageView.get());
		assetManager->writeAsset("pngWriteSuccessful.png", wp);
	}
	//TODO OBJ loader test 
	{
		//IAssetLoader::SAssetLoadParams lp;
		//auto bundle = assetManager->getAsset("../../media/sponza.obj", lp);
		//assert(!bundle.getContents().empty());
		//auto cpumesh = bundle.getContents().begin()[0];
		//auto cpumesh_raw = static_cast<ICPUMesh*>(cpumesh.get());
		//
		//IAssetWriter::SAssetWriteParams wp(cpumesh.get());
		//assetManager->writeAsset("objWriteSuccessful.obj", wp);
	}
	//JPEG loader test
	{
		IAssetLoader::SAssetLoadParams lp;
		auto asset = assetManager->getAsset("../../media/dwarf.jpg", lp);
		assert(!asset.getContents().empty());
		auto cpuImage = static_cast<ICPUImage*>(asset.getContents().begin()->get());
		core::smart_refctd_ptr<ICPUImageView> imageView;
		ICPUImageView::SCreationParams imgViewParams;
		imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		imgViewParams.format = E_FORMAT::EF_R8G8B8_SRGB;
		imgViewParams.image = core::smart_refctd_ptr<ICPUImage>(cpuImage);
		imgViewParams.viewType = ICPUImageView::ET_2D;
		imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
		imageView = ICPUImageView::create(std::move(imgViewParams));

		IAssetWriter::SAssetWriteParams wp(imageView.get());
		assetManager->writeAsset("jpgWriteSuccessful.jpg", wp);
	}
	while (true)
	{

	}
}
