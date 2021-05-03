// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/DebugDraw/CDraw3DLine.h"
#include "../common/CommonAPI.h"


#include "../common/QToQuitEventReceiver.h"

using namespace nbl;
using namespace core;

int main()
{
    constexpr uint32_t WIN_W = 1280;
    constexpr uint32_t WIN_H = 720;
    constexpr uint32_t SC_IMG_COUNT = 3u;

    auto initOutp = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "Draw3DLine");
    auto win = initOutp.window;
    auto gl = initOutp.apiConnection;
    auto surface = initOutp.surface;
    auto device = initOutp.logicalDevice;
    auto queue = initOutp.queue;
    auto sc = initOutp.swapchain;
    auto renderpass = initOutp.renderpass;
    auto fbo = initOutp.fbo;
    auto cmdpool = initOutp.commandPool;
    {
        video::IDriverMemoryBacked::SDriverMemoryRequirements mreq;
        core::smart_refctd_ptr<video::IGPUCommandBuffer> cb;
        device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cb);
        assert(cb);

        cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

        asset::SViewport vp;
        vp.minDepth = 1.f;
        vp.maxDepth = 0.f;
        vp.x = 0u;
        vp.y = 0u;
        vp.width = WIN_W;
        vp.height = WIN_H;
        cb->setViewport(0u, 1u, &vp);

        cb->end();

        video::IGPUQueue::SSubmitInfo info;
        auto* cb_ = cb.get();
        info.commandBufferCount = 1u;
        info.commandBuffers = &cb_;
        info.pSignalSemaphores = nullptr;
        info.signalSemaphoreCount = 0u;
        info.pWaitSemaphores = nullptr;
        info.waitSemaphoreCount = 0u;
        info.pWaitDstStageMask = nullptr;
        queue->submit(1u, &info, nullptr);
    }
    core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[SC_IMG_COUNT];
    device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT, cmdbuf);
    for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
    {
        auto& cb = cmdbuf[i];
        auto& fb = fbo[i];

        cb->begin(0);

        size_t offset = 0u;
        video::IGPUCommandBuffer::SRenderpassBeginInfo info;
        asset::SClearValue clear;
        asset::VkRect2D area;
        area.offset = { 0, 0 };
        area.extent = { WIN_W, WIN_H };
        clear.color.float32[0] = 1.f;
        clear.color.float32[1] = 1.f;
        clear.color.float32[2] = 1.f;
        clear.color.float32[3] = 1.f;
        info.renderpass = renderpass;
        info.framebuffer = fb;
        info.clearValueCount = 1u;
        info.clearValues = &clear;
        info.renderArea = area;
        cb->beginRenderPass(&info, asset::ESC_INLINE);
        cb->endRenderPass();

        cb->end();
    }

    //video::IVideoDriver* driver = device->getVideoDriver();
    //scene::ISceneManager* smgr = device->getSceneManager();
    auto draw3DLine = ext::DebugDraw::CDraw3DLine::create<SC_IMG_COUNT, WIN_W, WIN_H>(device.get(), queue, sc.get(), renderpass.get(), cmdbuf, fbo.data());

    //auto camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.001f);

    //camera->setPosition(core::vector3df(0,0,-10));
    //camera->setTarget(core::vector3df(0,0,0));
    //camera->setNearValue(0.01f);
    //camera->setFarValue(100.0f);

    //smgr->setActiveCamera(camera);

    uint64_t lastFPSTime = 0;

    core::vector<std::pair<ext::DebugDraw::S3DLineVertex, ext::DebugDraw::S3DLineVertex>> lines;

    for (int i = 0; i < 100; ++i)
    {
        lines.push_back({
        {
            { 0.f, 0.f, 0.f },     // start origin
            { 1.f, 0.f, 0.f, 1.f } // start color
        }, {
            { i % 2 ? float(i) : float(-i), 50.f, 10.f}, // end origin
            { 1.f, 0.f, 0.f, 1.f }         // end color
        }
        });
    }


  //  while(device->run() && receiver.keepOpen())
  //  if (device->isWindowActive())
  //  {
  //      driver->beginScene(true, true, video::SColor(255,255,255,255));

		//camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		//camera->render();
		//core::matrix4SIMD mvp = camera->getConcatenatedMatrix();

  //      draw3DLine->draw(mvp,
  //          0.f, 0.f, 0.f,   // start
  //          0.f, 100.f, 0.f, // end
  //          1.f, 0, 0, 1.f   // color
  //      );

  //      draw3DLine->draw(mvp,lines); // multiple lines

  //      driver->endScene();

  //      // display frames per second in window title
  //      uint64_t time = device->getTimer()->getRealTime();
  //      if (time-lastFPSTime > 1000)
  //      {
  //          std::wostringstream str(L"Draw3DLine Ext - Irrlicht Engine [");
  //          str.seekp(0,std::ios_base::end);
  //          str << driver->getName() << "] FPS:" << driver->getFPS();

  //          device->setWindowCaption(str.str());
  //          lastFPSTime = time;
  //      }
  //  }
    constexpr uint32_t FRAME_COUNT = 500000u;
    constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; //ns
    matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(90), float(WIN_W) / WIN_H, 0.01, 100);
    matrix3x4SIMD view = matrix3x4SIMD::buildCameraLookAtMatrixRH(core::vectorSIMDf(0, 0, -10), core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
    auto viewProj = matrix4SIMD::concatenateBFollowedByA(proj, matrix4SIMD(view));
    for (uint32_t i = 0u; i < FRAME_COUNT; ++i)
    {
        draw3DLine->draw(viewProj, lines);
    }

    device->waitIdle();

    return 0;
}


// If you see this line of code, i forgot to remove it
// It basically forces the usage of NVIDIA GPU
extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }