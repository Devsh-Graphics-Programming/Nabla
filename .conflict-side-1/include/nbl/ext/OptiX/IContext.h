// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_EXT_OPTIX_CONTEXT_H_INCLUDED__
#define __NBL_EXT_OPTIX_CONTEXT_H_INCLUDED__


#include "nabla.h"


#include "nbl/ext/OptiX/IModule.h"
#include "nbl/ext/OptiX/IDenoiser.h"

namespace nbl
{
namespace ext
{
namespace OptiX
{

class Manager;

#define _NBL_OPTIX_DEFAULT_NVRTC_OPTIONS "--std=c++14",cuda::CCUDAHandler::getCommonVirtualCUDAArchitecture(),"-dc","-use_fast_math","-default-device","--device-debug"

class IContext final : public core::IReferenceCounted
{
	public:
		inline OptixDeviceContext getOptiXHandle() {return optixContext;}


		// modules
		static const OptixModuleCompileOptions defaultOptixModuleCompileOptions;

		core::smart_refctd_ptr<IModule> createModuleFromPTX(const char* ptx, size_t ptx_size, const OptixPipelineCompileOptions& pipeline_options,
															const OptixModuleCompileOptions& module_options=defaultOptixModuleCompileOptions,
															std::string* log=nullptr);
		core::smart_refctd_ptr<IModule> createModuleFromPTX(const std::string& ptx, const OptixPipelineCompileOptions& pipeline_options,
															const OptixModuleCompileOptions& module_options=defaultOptixModuleCompileOptions,
															std::string* log=nullptr)
		{
			return createModuleFromPTX(ptx.c_str(),ptx.size(),pipeline_options,module_options,log);
		}

		template<typename CompileOptionsT = const std::initializer_list<const char*>&>
		inline core::smart_refctd_ptr<IModule> compileModuleFromSource(	const char* source, const char* filename, const OptixPipelineCompileOptions& pipeline_options,
																		const OptixModuleCompileOptions& module_options=defaultOptixModuleCompileOptions,
																		CompileOptionsT compile_options={_NBL_OPTIX_DEFAULT_NVRTC_OPTIONS}, std::string* log=nullptr)
		{
			return compileModuleFromSource_helper(source,filename,pipeline_options,module_options,&(*compile_options.begin()),&(*compile_options.end()),log);
		}

		template<typename CompileOptionsT = const std::initializer_list<const char*>&>
		inline core::smart_refctd_ptr<IModule> compileModuleFromFile(io::IReadFile* file, const OptixPipelineCompileOptions& pipeline_options,
																	const OptixModuleCompileOptions& module_options=defaultOptixModuleCompileOptions,
																	CompileOptionsT compile_options={_NBL_OPTIX_DEFAULT_NVRTC_OPTIONS}, std::string* log=nullptr)
		{
			if (!file)
				return nullptr;

			auto sz = file->getSize();
			core::vector<char> tmp(sz+1ull);
			file->read(tmp.data(),sz);
			tmp.back() = 0;
			return compileModuleFromSource<CompileOptionsT>(tmp.data(),file->getFileName().c_str(),pipeline_options,module_options,std::forward<CompileOptionsT>(compile_options),log);
		}

		core::smart_refctd_ptr<IDenoiser> createDenoiser(const OptixDenoiserOptions* options, OptixDenoiserModelKind model=OPTIX_DENOISER_MODEL_KIND_HDR, void* modelData=nullptr, size_t modelDataSizeInBytes=0ull)
		{
			if (!options)
				return nullptr;

			OptixDenoiser denoiser = nullptr;
			if (optixDenoiserCreate(optixContext,options,&denoiser)!=OPTIX_SUCCESS || !denoiser)
				return nullptr;

			auto denoiser_wrapper = core::smart_refctd_ptr<IDenoiser>(new IDenoiser(denoiser),core::dont_grab);
			if (optixDenoiserSetModel(denoiser,model,modelData,modelDataSizeInBytes)!=OPTIX_SUCCESS)
				return nullptr;

			return denoiser_wrapper;
		}

	protected:
		friend class Manager;

		IContext(Manager* _manager, CUcontext _CUDAContext, OptixDeviceContext _optixContext) : manager(_manager), CUDAContext(_CUDAContext), optixContext(_optixContext)
		{
			assert(manager && CUDAContext && optixContext);
		}
		~IContext()
		{
			if (cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuCtxPushCurrent_v2(CUDAContext)))
			{
				CUcontext tmp = nullptr;
				cuda::CCUDAHandler::cuda.pcuCtxSynchronize();
				cuda::CCUDAHandler::cuda.pcuCtxPopCurrent_v2(&tmp);
			}
			
			optixDeviceContextDestroy(optixContext);
		}

		core::smart_refctd_ptr<IModule> compileModuleFromSource_helper(	const char* source, const char* filename,
																		const OptixPipelineCompileOptions& pipeline_options,
																		const OptixModuleCompileOptions& module_options,
																		const char* const* compile_options_begin, const char* const* compile_options_end,
																		std::string* log);


		
		Manager* manager;
		CUcontext CUDAContext;
		OptixDeviceContext optixContext;
};

}
}
}

#endif