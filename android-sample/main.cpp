/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

//BEGIN_INCLUDE(all)
#include <initializer_list>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <cerrno>
#include <cassert>
#include <nabla.h>
#include <nbl/ui/CWindowAndroid.h>

//#include <EGL/egl.h>
//#include <GLES/gl.h>

#include <android/sensor.h>
#include <android/log.h>
#include <android_native_app_glue.h>
//#include "debugbreak.h"

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

using namespace nbl;

static constexpr uint32_t SC_IMG_COUNT = 3u;

/**
 * Our saved state data.
 */
struct saved_state {
    float angle;
    int32_t x;
    int32_t y;
};

struct nabla {
    struct android_app* app;

    ASensorManager* sensorManager;
    const ASensor* accelerometerSensor;
    ASensorEventQueue* sensorEventQueue;

    core::smart_refctd_ptr<ui::IWindow> window;
    core::smart_refctd_ptr<system::ISystem> system;
    core::smart_refctd_ptr<video::IAPIConnection> api;
    core::smart_refctd_ptr<video::IPhysicalDevice> gpu;
    core::smart_refctd_ptr<video::ILogicalDevice> dev;
    core::smart_refctd_ptr<video::ISwapchain> sc;
    core::smart_refctd_ptr<video::IGPURenderpass> renderpass;
    core::smart_refctd_ptr<video::IGPUFramebuffer> fbo[SC_IMG_COUNT];
    core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
    core::smart_refctd_ptr<video::IGPUBuffer> buffer;
    core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[SC_IMG_COUNT];

    struct saved_state state;
};

/**
 * Initialize an EGL context for the current display.
 */
static int engine_init_display(struct nabla* engine) {
    //debug_break();
    // initialize OpenGL ES and EGL
    engine->system = core::make_smart_refctd_ptr<system::ISystem>(nullptr);
    engine->window = core::make_smart_refctd_ptr<ui::CWindowAndroid>(engine->app->window);

    COpenGLDebugCallback cb;
    engine->api = video::COpenGLConection::create(core::smart_refctd_ptr<system::ISystem>(engine->system), video::EAT_OPENGL_ES, 0, "android-sample", std::move(cb));

    auto surface = video::CSurfaceGLAndroid::create(core::smart_refctd_ptr(engine->api),engine->window));

    auto gpus = engine->api->getPhysicalDevices();
	assert(!gpus.empty());
    engine->gpu = gpus.begin()[0];

    assert(surface->isSupported(engine->gpu.get(), 0u));
    
    video::ILogicalDevice::SCreationParams dev_params;
	dev_params.queueParamsCount = 1u;
	video::ILogicalDevice::SQueueCreationParams q_params;
	q_params.familyIndex = 0u;
	q_params.count = 1u;//4u;
	q_params.flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
	float priority[4] = {1.f,1.f,1.f,1.f};
	q_params.priorities = priority;
	dev_params.queueCreateInfos = &q_params;
	engine->dev = engine->gpu->createLogicalDevice(dev_params);

    auto device = engine->dev;
    auto* queue = device->getQueue(0u, 0u);

    const uint32_t win_w = engine->window->getWidth();
    const uint32_t win_h = engine->window->getHeight();

    {
		video::ISwapchain::SCreationParams sc_params;
		sc_params.width = win_w;
		sc_params.height = win_h;
		sc_params.arrayLayers = 1u;
		sc_params.minImageCount = SC_IMG_COUNT;
		sc_params.presentMode = video::ISurface::EPM_FIFO_RELAXED;
		sc_params.surface = surface;
		sc_params.surfaceFormat.format = asset::EF_R8G8B8A8_SRGB;
		sc_params.surfaceFormat.colorSpace.eotf = asset::EOTF_sRGB;
		sc_params.surfaceFormat.colorSpace.primary = asset::ECP_SRGB;

		engine->sc = device->createSwapchain(std::move(sc_params));
		assert(engine->sc);
	}

    {
		video::IGPURenderpass::SCreationParams::SAttachmentDescription a;
		a.initialLayout = asset::EIL_UNDEFINED;
		a.finalLayout = asset::EIL_UNDEFINED;
		a.format = asset::EF_R8G8B8A8_SRGB;
		a.samples = asset::IImage::ESCF_1_BIT;
		a.loadOp = video::IGPURenderpass::ELO_CLEAR;
		a.storeOp = video::IGPURenderpass::ESO_STORE;

		video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
		colorAttRef.attachment = 0u;
		colorAttRef.layout = asset::EIL_UNDEFINED;
		video::IGPURenderpass::SCreationParams::SSubpassDescription sp;
		sp.colorAttachmentCount = 1u;
		sp.colorAttachments = &colorAttRef;
		sp.depthStencilAttachment = nullptr;
		sp.flags = video::IGPURenderpass::ESDF_NONE;
		sp.inputAttachmentCount = 0u;
		sp.inputAttachments = nullptr;
		sp.preserveAttachmentCount = 0u;
		sp.preserveAttachments = nullptr;
		sp.resolveAttachments = nullptr;

		video::IGPURenderpass::SCreationParams rp_params;
		rp_params.attachmentCount = 1u;
		rp_params.attachments = &a;
		rp_params.dependencies = nullptr;
		rp_params.dependencyCount = 0u;
		rp_params.subpasses = &sp;
		rp_params.subpassCount = 1u;

		engine->renderpass = device->createGPURenderpass(rp_params);
	}
    auto renderpass = engine->renderpass;

    auto sc_images = engine->sc->getImages();

    for (uint32_t i = 0u; i < sc_images.size(); ++i)
	{
		auto img = sc_images.begin()[i];
		core::smart_refctd_ptr<video::IGPUImageView> view;
		{
			video::IGPUImageView::SCreationParams view_params;
			view_params.format = img->getCreationParameters().format;
			view_params.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
			view_params.subresourceRange.baseMipLevel = 0u;
			view_params.subresourceRange.levelCount = 1u;
			view_params.subresourceRange.baseArrayLayer = 0u;
			view_params.subresourceRange.layerCount = 1u;
			view_params.image = std::move(img);

			view = device->createGPUImageView(std::move(view_params));
			assert(view);
		}

		video::IGPUFramebuffer::SCreationParams fb_params;
		fb_params.width = win_w;
		fb_params.height = win_h;
		fb_params.layers = 1u;
		fb_params.renderpass = engine->renderpass;
		fb_params.flags = static_cast<video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
		fb_params.attachmentCount = 1u;
		fb_params.attachments = &view;

		engine->fbo[i] = device->createGPUFramebuffer(std::move(fb_params));
		assert(engine->fbo[i]);
	}

    auto cmdpool = device->createCommandPool(0u, static_cast<video::IGPUCommandPool::E_CREATE_FLAGS>(0));
	assert(cmdpool);

#include "nbl/nblpack.h"
	struct SVertex
	{
		float pos[2];
		float color[3];
	} PACK_STRUCT;
#include "nbl/nblunpack.h"

    auto layout = device->createGPUPipelineLayout();
	assert(layout);

	core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpindependent_pipeline;
	{
        const char* vs_source = R"(#version 430

layout (location = 0) in vec2 Pos;
layout (location = 1) in vec3 Color;

layout (location = 0) out vec3 OutColor;

void main()
{
    OutColor = Color;
    gl_Position = vec4(Pos, 0.0, 1.0);
}
)";
        const char* fs_source = R"(#version 430

layout (location = 0) in vec3 InColor;
layout (location = 0) out vec4 OutColor;

void main()
{
    OutColor = vec4(InColor, 1.0);
}
)";

		auto vs_unspec = device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(vs_source));
		auto fs_unspec = device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(fs_source));

		asset::ISpecializedShader::SInfo vsinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_VERTEX, "vs");
		auto vs = device->createGPUSpecializedShader(vs_unspec.get(), vsinfo);
		asset::ISpecializedShader::SInfo fsinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_FRAGMENT, "fs");
		auto fs = device->createGPUSpecializedShader(fs_unspec.get(), fsinfo);

		video::IGPUSpecializedShader* shaders[2]{ vs.get(), fs.get() };

		asset::SVertexInputParams vtxinput;
		vtxinput.attributes[0].binding = 0;
		vtxinput.attributes[0].format = asset::EF_R32G32_SFLOAT;
		vtxinput.attributes[0].relativeOffset = offsetof(SVertex, pos);

		vtxinput.attributes[1].binding = 0;
		vtxinput.attributes[1].format = asset::EF_R32G32B32_SFLOAT;
		vtxinput.attributes[1].relativeOffset = offsetof(SVertex, color);

		vtxinput.bindings[0].inputRate = asset::EVIR_PER_VERTEX;
		vtxinput.bindings[0].stride = sizeof(SVertex);

		vtxinput.enabledAttribFlags = 0b0011;
		vtxinput.enabledBindingFlags = 0b0001;
		
		asset::SRasterizationParams raster;
		raster.depthTestEnable = 0;
		raster.depthWriteEnable = 0;
		raster.faceCullingMode = asset::EFCM_NONE;
		
		asset::SPrimitiveAssemblyParams primitive;
		primitive.primitiveType = asset::EPT_TRIANGLE_LIST;

		asset::SBlendParams blend;

		rpindependent_pipeline = device->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(layout), shaders, shaders+2, vtxinput, blend, primitive, raster);
		assert(rpindependent_pipeline);
	}

	{
		video::IGPUGraphicsPipeline::SCreationParams gp_params;
		gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
		gp_params.renderpass = renderpass;
		gp_params.renderpassIndependent = rpindependent_pipeline;
		gp_params.subpassIx = 0u;

		engine->pipeline = device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));
	}

    {
		const SVertex vertices[3]{
			{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
			{{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
			{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
		};

		video::IDriverMemoryBacked::SDriverMemoryRequirements mreq;
		
		auto mreqs = device->getDeviceLocalGPUMemoryReqs();
		mreqs.vulkanReqs.size = sizeof(vertices);
		engine->buffer = device->createGPUBufferOnDedMem(mreqs, true);
		assert(engine->buffer);
        auto buffer = engine->buffer;

		core::smart_refctd_ptr<video::IGPUCommandBuffer> cb;
		device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cb);
		assert(cb);

		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = win_w;
		vp.height = win_h;
		cb->setViewport(0u, 1u, &vp);

		cb->updateBuffer(buffer.get(), 0u, sizeof(vertices), vertices);

		video::IGPUCommandBuffer::SBufferMemoryBarrier bufMemBarrier;
		bufMemBarrier.srcQueueFamilyIndex = 0u;
		bufMemBarrier.dstQueueFamilyIndex = 0u;
		bufMemBarrier.offset = 0u;
		bufMemBarrier.size = buffer->getSize();
		bufMemBarrier.buffer = buffer;
		bufMemBarrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
		bufMemBarrier.barrier.dstAccessMask = asset::EAF_VERTEX_ATTRIBUTE_READ_BIT;
		cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_VERTEX_INPUT_BIT, 0, 0u, nullptr, 1u, &bufMemBarrier, 0u, nullptr);

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

    device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT, engine->cmdbuf);
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		auto& cb = engine->cmdbuf[i];
		auto& fb = engine->fbo[i];

		cb->begin(0);
		
		auto* buf = engine->buffer.get();
		size_t offset = 0u;
		cb->bindVertexBuffers(0u, 1u, &buf, &offset);
		cb->bindGraphicsPipeline(engine->pipeline.get());
		video::IGPUCommandBuffer::SRenderpassBeginInfo info;
		asset::SClearValue clear;
		VkRect2D area;
		area.offset = { 0, 0 };
		area.extent = { win_w, win_h };
		clear.color.float32[0] = 0.f;
		clear.color.float32[1] = 1.f;
		clear.color.float32[2] = 1.f;
		clear.color.float32[3] = 1.f;
		info.renderpass = engine->renderpass;
		info.framebuffer = fb;
		info.clearValueCount = 1u;
		info.clearValues = &clear;
		info.renderArea = area;
		cb->beginRenderPass(&info, asset::ESC_INLINE);
		cb->draw(3u, 1u, 0u, 0u);
		cb->endRenderPass();

		cb->end();
	}

    return 0;
}

/**
 * Just the current frame in the display.
 */
static void engine_draw_frame(struct nabla* engine) {
    if (!engine->dev) {
        return;
    }

    constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; //ns

    auto img_acq_sem = engine->dev->createSemaphore();
    auto render_finished_sem = engine->dev->createSemaphore();

    auto* queue = engine->dev->getQueue(0u, 0u);

    uint32_t imgnum = 0u;
    engine->sc->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

    video::IGPUQueue::SSubmitInfo submit;
    {
        auto* cb = engine->cmdbuf[imgnum].get();
        submit.commandBufferCount = 1u;
        submit.commandBuffers = &cb;
        video::IGPUSemaphore* signalsem = render_finished_sem.get();
        submit.signalSemaphoreCount = 1u;
        submit.pSignalSemaphores = &signalsem;
        video::IGPUSemaphore* waitsem = img_acq_sem.get();
        asset::E_PIPELINE_STAGE_FLAGS dstWait = asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;
        submit.waitSemaphoreCount = 1u;
        submit.pWaitSemaphores = &waitsem;
        submit.pWaitDstStageMask = &dstWait;

        queue->submit(1u, &submit, nullptr);
    }

    video::IGPUQueue::SPresentInfo present;
    {
        present.swapchainCount = 1u;
        present.imgIndices = &imgnum;
        video::ISwapchain* swapchain = engine->sc.get();
        present.swapchains = &swapchain;
        video::IGPUSemaphore* waitsem = render_finished_sem.get();
        present.waitSemaphoreCount = 1u;
        present.waitSemaphores = &waitsem;

        queue->present(present);
    }
}

/**
 * Tear down the EGL context currently associated with the display.
 */
static void engine_term_display(struct nabla* engine) {
    //debug_break();

    engine->dev->waitIdle();

    for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
        engine->cmdbuf[i] = nullptr;
    engine->buffer = nullptr;
    engine->pipeline = nullptr;
    for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
        engine->fbo[i] = nullptr;
    engine->renderpass = nullptr;
    engine->sc = nullptr;
    engine->dev = nullptr;
    engine->gpu = nullptr;
    engine->api = nullptr;
    engine->window = nullptr;
}

/**
 * Process the next input event.
 */
static int32_t engine_handle_input(struct android_app* app, AInputEvent* event) {
    nabla* engine = (struct nabla*)app->userData;
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
        engine->state.x = AMotionEvent_getX(event, 0);
        engine->state.y = AMotionEvent_getY(event, 0);
        return 1;
    }
    return 0;
}

/**
 * Process the next main command.
 */
static void engine_handle_cmd(struct android_app* app, int32_t cmd) {
    //debug_break();
    auto* engine = (struct nabla*)app->userData;
    switch (cmd) {
        case APP_CMD_SAVE_STATE:
            // The system has asked us to save our current state.  Do so.
            engine->app->savedState = malloc(sizeof(struct saved_state));
            *((struct saved_state*)engine->app->savedState) = engine->state;
            engine->app->savedStateSize = sizeof(struct saved_state);
            break;
        case APP_CMD_INIT_WINDOW:
            //debug_break();
            // The window is being shown, get it ready.
            if (engine->app->window != nullptr) {
                engine_init_display(engine);
                engine_draw_frame(engine);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            // The window is being hidden or closed, clean it up.
            engine_term_display(engine);
            break;
        /*
        case APP_CMD_GAINED_FOCUS:
            // When our app gains focus, we start monitoring the accelerometer.
            if (engine->accelerometerSensor != nullptr) {
                ASensorEventQueue_enableSensor(engine->sensorEventQueue,
                                               engine->accelerometerSensor);
                // We'd like to get 60 events per second (in us).
                ASensorEventQueue_setEventRate(engine->sensorEventQueue,
                                               engine->accelerometerSensor,
                                               (1000L/60)*1000);
            }
            break;
        case APP_CMD_LOST_FOCUS:
            // When our app loses focus, we stop monitoring the accelerometer.
            // This is to avoid consuming battery while not being used.
            if (engine->accelerometerSensor != nullptr) {
                ASensorEventQueue_disableSensor(engine->sensorEventQueue,
                                                engine->accelerometerSensor);
            }
            // Also stop animating.
            engine->animating = 0;
            engine_draw_frame(engine);
            break;
        */
        default:
            break;
    }
}

/*
 * AcquireASensorManagerInstance(void)
 *    Workaround ASensorManager_getInstance() deprecation false alarm
 *    for Android-N and before, when compiling with NDK-r15
 */
#include <dlfcn.h>
ASensorManager* AcquireASensorManagerInstance(android_app* app) {

  if(!app)
    return nullptr;

  typedef ASensorManager *(*PF_GETINSTANCEFORPACKAGE)(const char *name);
  void* androidHandle = dlopen("libandroid.so", RTLD_NOW);
  auto getInstanceForPackageFunc = (PF_GETINSTANCEFORPACKAGE)
      dlsym(androidHandle, "ASensorManager_getInstanceForPackage");
  if (getInstanceForPackageFunc) {
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);

    jclass android_content_Context = env->GetObjectClass(app->activity->clazz);
    jmethodID midGetPackageName = env->GetMethodID(android_content_Context,
                                                   "getPackageName",
                                                   "()Ljava/lang/String;");
    auto packageName= (jstring)env->CallObjectMethod(app->activity->clazz,
                                                        midGetPackageName);

    const char *nativePackageName = env->GetStringUTFChars(packageName, nullptr);
    ASensorManager* mgr = getInstanceForPackageFunc(nativePackageName);
    env->ReleaseStringUTFChars(packageName, nativePackageName);
    app->activity->vm->DetachCurrentThread();
    if (mgr) {
      dlclose(androidHandle);
      return mgr;
    }
  }

  typedef ASensorManager *(*PF_GETINSTANCE)();
  auto getInstanceFunc = (PF_GETINSTANCE)
      dlsym(androidHandle, "ASensorManager_getInstance");
  // by all means at this point, ASensorManager_getInstance should be available
  assert(getInstanceFunc);
  dlclose(androidHandle);

  return getInstanceFunc();
}


/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(struct android_app* state) {
    struct nabla engine{};

    state->userData = &engine;
    state->onAppCmd = engine_handle_cmd;
    state->onInputEvent = engine_handle_input;
    engine.app = state;

    //debug_break();

    LOGI("Entered main!");

/*
    // Prepare to monitor accelerometer
    engine.sensorManager = AcquireASensorManagerInstance(state);
    engine.accelerometerSensor = ASensorManager_getDefaultSensor(
                                        engine.sensorManager,
                                        ASENSOR_TYPE_ACCELEROMETER);
    engine.sensorEventQueue = ASensorManager_createEventQueue(
                                    engine.sensorManager,
                                    state->looper, LOOPER_ID_USER,
                                    nullptr, nullptr);
*/

    if (state->savedState != nullptr) {
        // We are starting with a previous saved state; restore from it.
        engine.state = *(struct saved_state*)state->savedState;
    }

    // loop waiting for stuff to do.

    while (true) {
        // Read all pending events.
        int ident;
        int events;
        struct android_poll_source* source;

        LOGI("Entered main loop iteration!");
        //debug_break();

        // If not animating, we will block forever waiting for events.
        // If animating, we loop until all events are read, then continue
        // to draw the next frame of animation.
        while ((ident = ALooper_pollAll(0, nullptr, &events, (void**)&source)) >= 0) {

            LOGI("Entered poll loop iteration!");

            // Process this event.
            if (source != nullptr) {
                source->process(state, source);
            }

            // If a sensor has data, process it now.
            /*
            if (ident == LOOPER_ID_USER) {
                if (engine.accelerometerSensor != nullptr) {
                    ASensorEvent event;
                    while (ASensorEventQueue_getEvents(engine.sensorEventQueue,
                                                       &event, 1) > 0) {
                        LOGI("accelerometer: x=%f y=%f z=%f",
                             event.acceleration.x, event.acceleration.y,
                             event.acceleration.z);
                    }
                }
            }
            */

            // Check if we are exiting.
            if (state->destroyRequested != 0) {
                engine_term_display(&engine);
                return;
            }
        }

        engine_draw_frame(&engine);
    }
}
//END_INCLUDE(all)
