// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>


/**
This example just shows a screen which clears to red,
nothing fancy, just to show that Irrlicht links fine
**/
using namespace nbl;


/*
The start of the main function starts like in most other example. We ask the
user for the desired renderer and start it up.
*/
int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = core::dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	params.AuxGLContexts = 16;
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.

	constexpr uint32_t MaxSLI = 4u;
	uint32_t contextCount;
	CUcontext context[MaxSLI];
	bool ownContext[MaxSLI];
	CUstream stream[MaxSLI];
	{
		cuda::CCUDAHandler::init();

		int32_t version = 0;
		cuda::CCUDAHandler::cuda.pcuDriverGetVersion(&version);
		if (version < 7000)
			return 2;
		
		int nvrtcVersion[2] = {-1,-1};
		cuda::CCUDAHandler::nvrtc.pnvrtcVersion(nvrtcVersion+0,nvrtcVersion+1);
		if (nvrtcVersion[0]<7)
			return 3;

		// find device
		uint32_t foundDeviceCount = 0u;
		CUdevice devices[MaxSLI] = {};
		cuda::CCUDAHandler::getDefaultGLDevices(&foundDeviceCount, devices, MaxSLI);

		// create context
		CUcontext contexts[MaxSLI] = {};
		bool ownContexts[MaxSLI] = {};
		uint32_t suitableDevices = 0u;
		for (uint32_t i = 0u; i < foundDeviceCount; i++)
		{
			if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuCtxCreate_v2(contexts + suitableDevices, CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST | CU_CTX_LMEM_RESIZE_TO_MAX, devices[suitableDevices])))
				continue;

			uint32_t version = 0u;
			cuda::CCUDAHandler::cuda.pcuCtxGetApiVersion(contexts[suitableDevices], &version);
			if (version < 3020)
			{
				cuda::CCUDAHandler::cuda.pcuCtxDestroy_v2(contexts[suitableDevices]);
				continue;
			}
			cuda::CCUDAHandler::cuda.pcuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
			ownContexts[suitableDevices++] = true;
		}

		if (!suitableDevices)
			return 4;
		
		contextCount = suitableDevices;
		for (uint32_t i=0u; i< contextCount; i++)
		{
			context[i] = contexts[i];
			ownContext[i] = ownContexts ? ownContexts[i]:false;
			if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuStreamCreate(stream+i,CU_STREAM_NON_BLOCKING)))
			{
				i--;
				contextCount--;
				continue;
			}

			//OptixDeviceContextOptions options = {};
			//optixDeviceContextCreate(context[i], &options, optixContext);
		}
	}


	nvrtcProgram program = nullptr;
	// load program from file
	auto file = device->getFileSystem()->createAndOpenFile("../vectorAdd_kernel.cu");
	if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::createProgram<io::IReadFile*>(&program, file, nullptr, nullptr)))
		return 5;
	file->drop();

	if (!program)
		return 6;

	if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::compileProgram(program)))
		return 7;

	std::string log;
	cuda::CCUDAHandler::getProgramLog(program,log);
	printf("NVRTC Compile Log:\n%s\n", log.c_str());

	std::string ptx;
	if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::getPTX(program,ptx)))
		return 8;

	cuda::CCUDAHandler::nvrtc.pnvrtcDestroyProgram(&program);

	CUmodule module;
	if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuModuleLoadDataEx(&module,ptx.c_str(),0u,nullptr,nullptr)))
		return 9;
	
	CUfunction kernel;
	if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuModuleGetFunction(&kernel, module, "vectorAdd")))
		return 10;



	constexpr uint32_t gridDim[3] = { 4096,1,1 };
	constexpr uint32_t blockDim[3] = { 1024,1,1 };
	int numElements = gridDim[0]*blockDim[0];
	auto _size = sizeof(float)*numElements;

	video::IVideoDriver* driver = device->getVideoDriver();


	core::smart_refctd_ptr<asset::ICPUBuffer> cpubuffers[2] = {	core::make_smart_refctd_ptr<asset::ICPUBuffer>(_size),
																core::make_smart_refctd_ptr<asset::ICPUBuffer>(_size)};
	for (auto j=0; j<2; j++)
	for (auto i=0; i<numElements; i++)
		reinterpret_cast<float*>(cpubuffers[j]->getPointer())[i] = rand();

	{
		constexpr auto resourceCount = 3u;
		cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer> resources[resourceCount];
		auto& A = resources[0] = driver->createFilledDeviceLocalBufferOnDedMem(_size,cpubuffers[0]->getPointer());
		auto& B = resources[1] = driver->createFilledDeviceLocalBufferOnDedMem(_size,cpubuffers[1]->getPointer());
		auto& C = resources[2] = driver->createDeviceLocalGPUBufferOnDedMem(_size);
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&A,CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)))
			return 11;
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&B,CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)))
			return 11;
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&C,CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD)))
			return 12;

		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::acquireAndGetPointers(resources,resources+resourceCount,stream[0])))
			return 13;

		void* parameters[] = {&A.asBuffer.pointer,&B.asBuffer.pointer,&C.asBuffer.pointer,&numElements};
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuLaunchKernel(kernel,gridDim[0],gridDim[1],gridDim[2],
																									blockDim[0],blockDim[1],blockDim[2],
																									0,stream[0],parameters,nullptr)))
			return 14;

		auto scratch = parameters;
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::releaseResourcesToGraphics(scratch,resources,resources+resourceCount,stream[0])))
			return 15;

		float* C_host = new float[numElements];
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuMemcpyDtoHAsync_v2(C_host,C.asBuffer.pointer,_size,stream[0])))
			return 16;
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuCtxSynchronize()))
			return 17;

		// check
		for (auto i = 0; i < numElements; i++)
			assert(abs(C_host[i] - reinterpret_cast<float*>(cpubuffers[0]->getPointer())[i] - reinterpret_cast<float*>(cpubuffers[1]->getPointer())[i]) < 0.01f);

		delete[] C_host;
	}


	if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuModuleUnload(module)))
		return 19;
	
	for (uint32_t i=0u; i<contextCount; i++)
	{
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuCtxPushCurrent_v2(context[i])))
			continue;
		cuda::CCUDAHandler::cuda.pcuCtxSynchronize();

		cuda::CCUDAHandler::cuda.pcuStreamDestroy_v2(stream[i]);
		if (ownContext[i])
			cuda::CCUDAHandler::cuda.pcuCtxDestroy_v2(context[i]);
	}

	return 0;
}
