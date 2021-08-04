// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>
// TODO: get all these headers into "nabla.h"
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

class WindowEventCallback;

class InputSystem : public IReferenceCounted
{
	public:
		template <class ChannelType>
		struct Channels
		{
			core::mutex lock;
			std::condition_variable added;
			core::vector<core::smart_refctd_ptr<ChannelType>> channels;
		};
		// TODO: move to "nbl/ui/InputEventChannel.h" once the interface of this utility struct matures, also maybe rename to `Consumer` ?
		template <class ChannelType>
		struct ChannelReader
		{
			template<typename F>
			inline void consumeEvents(F&& processFunc, system::logger_opt_ptr logger=nullptr)
			{
				auto events = channel->getEvents();
				const auto frontBufferCapacity = channel->getFrontBufferCapacity();
				if (events.size()>consumedCounter+frontBufferCapacity)
				{
					logger.log(
						"Detected overflow, %d unconsumed events in channel of size %d!",
						system::ILogger::ELL_ERROR,events.size()-consumedCounter,frontBufferCapacity
					);
					consumedCounter = events.size()-frontBufferCapacity;
				}
				processFunc(ChannelType::range_t(events.begin()+consumedCounter,events.end()));
				consumedCounter = events.size();
			}

			core::smart_refctd_ptr<ChannelType> channel = nullptr;
			uint64_t consumedCounter = 0ull;
		};
		
		InputSystem(system::logger_opt_smart_ptr&& logger) : m_logger(std::move(logger)) {}

		void getDefaultMouse(ChannelReader<IMouseEventChannel>* reader)
		{
			getDefault(m_mouse,reader);
		}
		void getDefaultKeyboard(ChannelReader<IKeyboardEventChannel>* reader)
		{
			getDefault(m_keyboard,reader);
		}

	private:
		friend class WindowEventCallback;
		template<class ChannelType>
		void add(Channels<ChannelType>& channels, core::smart_refctd_ptr<ChannelType>&& channel)
		{
			std::unique_lock lock(channels.lock);
			channels.channels.push_back(std::move(channel));
			channels.added.notify_all();
		}
		template<class ChannelType>
		void remove(Channels<ChannelType>& channels, const ChannelType* channel)
		{
			std::unique_lock lock(channels.lock);
			channels.channels.erase(
				std::find_if(
					channels.channels.begin(),channels.channels.end(),[channel](const auto& chan)->bool{return chan.get()==channel;}
				)
			);
		}
		template<class ChannelType>
		void getDefault(Channels<ChannelType>& channels, ChannelReader<ChannelType>* reader)
		{
			/*
			* TODO: Improve default device switching.
			* For nice results, we should actually make a multi-channel reader,
			* and then keep a consumed counter together with a last consumed event from each channel.
			* If there is considerable pause in events received by our current chosen channel or
			* we can detect some other channel of the same "compatible class" is producing more events,
			* Switch the channel choice, but prune away all events younger than the old default's consumption timestamp.
			* (Basically switch keyboards but dont try to process events older than the events you've processed from the old keyboard)
			*/
			std::unique_lock lock(channels.lock);
			while (channels.channels.empty())
			{
				m_logger.log("Waiting For Input Device to be connected...",system::ILogger::ELL_INFO);
				channels.added.wait(lock);
			}

			auto current_default = channels.channels.front();
			if (reader->channel==current_default)
				return;

			reader->channel = current_default;
			reader->consumedCounter = 0u;
		}

		system::logger_opt_smart_ptr m_logger;
		Channels<IMouseEventChannel> m_mouse;
		Channels<IKeyboardEventChannel> m_keyboard;
};

class WindowEventCallback : public IWindow::IEventCallback
{
public:
	WindowEventCallback(core::smart_refctd_ptr<InputSystem>&& inputSystem, system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(inputSystem)), m_logger(std::move(logger)) {}

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
		m_logger.log("A mouse %p has been connected", system::ILogger::ELL_INFO, mch);
		m_inputSystem.get()->add(m_inputSystem.get()->m_mouse,std::move(mch));
	}
	void onMouseDisconnected_impl(IMouseEventChannel* mch) override
	{
		m_logger.log("A mouse %p has been disconnected", system::ILogger::ELL_INFO, mch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse,mch);
	}
	void onKeyboardConnected_impl(core::smart_refctd_ptr<IKeyboardEventChannel>&& kbch) override
	{
		m_logger.log("A keyboard %p has been connected", system::ILogger::ELL_INFO, kbch);
		m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard,std::move(kbch));
	}
	void onKeyboardDisconnected_impl(IKeyboardEventChannel* kbch) override
	{
		m_logger.log("A keyboard %p has been disconnected", system::ILogger::ELL_INFO, kbch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard,kbch);
	}

private:
	core::smart_refctd_ptr<InputSystem> m_inputSystem;
	system::logger_opt_smart_ptr m_logger;
};

int main()
{
	auto system = CommonAPI::createSystem();
	// *** Select stdout/file logger ***
	auto logger = make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>();
	//auto logger = system::CFileLogger::create(logFileName);
	// *** If you don't want logging, uncomment this one line***
	// logger = nullptr;
	// **************************************************************************************

	auto assetManager = core::make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(system));

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

	auto input = make_smart_refctd_ptr<InputSystem>(system::logger_opt_smart_ptr(logger));
	auto windowCb = make_smart_refctd_ptr<WindowEventCallback>(core::smart_refctd_ptr(input),system::logger_opt_smart_ptr(logger));
	params.callback = windowCb;
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

	// polling for events!
	InputSystem::ChannelReader<IMouseEventChannel> mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	auto mouseProcess = [logger](const IMouseEventChannel::range_t& events) -> void
	{
		for (auto eventIt=events.begin(); eventIt!=events.end(); eventIt++)
		{
			logger->log("Mouse event at %d us",system::ILogger::ELL_INFO,(*eventIt).timeStamp);
		}
	};
	auto keyboardProcess = [logger](const IKeyboardEventChannel::range_t& events) -> void
	{
		for (auto eventIt=events.begin(); eventIt!=events.end(); eventIt++)
		{
			logger->log("Keyboard event at %d us",system::ILogger::ELL_INFO,(*eventIt).timeStamp);
		}
	};
	
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
		input->getDefaultMouse(&mouse);
		input->getDefaultKeyboard(&keyboard);

		mouse.consumeEvents(mouseProcess,logger.get());
		keyboard.consumeEvents(keyboardProcess,logger.get());
	}
}
