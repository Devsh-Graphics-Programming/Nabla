// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_DEVICE_H_
#define _NBL_VIDEO_C_CUDA_DEVICE_H_


#include "nbl/video/IPhysicalDevice.h"


#ifdef _NBL_COMPILE_WITH_CUDA_

#include "cuda.h"
#include "nvrtc.h"
#if CUDA_VERSION < 9000
	#error "Need CUDA 9.0 SDK or higher."
#endif

// useful includes in the future
//#include "cudaEGL.h"
//#include "cudaVDPAU.h"

namespace nbl::video
{
class CCUDAHandler;
class CCUDASharedMemory;
class CCUDASharedSemaphore;

class CCUDADevice : public core::IReferenceCounted
{
    public:
		enum E_VIRTUAL_ARCHITECTURE
		{
			EVA_30,
			EVA_32,
			EVA_35,
			EVA_37,
			EVA_50,
			EVA_52,
			EVA_53,
			EVA_60,
			EVA_61,
			EVA_62,
			EVA_70,
			EVA_72,
			EVA_75,
			EVA_80,
			EVA_COUNT
		};
		static inline constexpr const char* virtualArchCompileOption[] = {
			"-arch=compute_30",
			"-arch=compute_32",
			"-arch=compute_35",
			"-arch=compute_37",
			"-arch=compute_50",
			"-arch=compute_52",
			"-arch=compute_53",
			"-arch=compute_60",
			"-arch=compute_61",
			"-arch=compute_62",
			"-arch=compute_70",
			"-arch=compute_72",
			"-arch=compute_75",
			"-arch=compute_80"
		};
		inline E_VIRTUAL_ARCHITECTURE getVirtualArchitecture() {return m_virtualArchitecture;}

		inline core::SRange<const char* const> geDefaultCompileOptions() const
		{
			return {m_defaultCompileOptions.data(),m_defaultCompileOptions.data()+m_defaultCompileOptions.size()};
		}

		// TODO/REDO Vulkan: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html
		// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vulkan-interoperability
		// Watch out, use Driver API (`cu` functions) NOT the Runtime API (`cuda` functions)
		// Also maybe separate this out into its own `CCUDA` class instead of nesting it here?
#if 0
		template<typename ObjType>
		struct GraphicsAPIObjLink
		{
				GraphicsAPIObjLink() : obj(nullptr), cudaHandle(nullptr), acquired(false)
				{
					asImage = {nullptr};
				}
				GraphicsAPIObjLink(core::smart_refctd_ptr<ObjType>&& _obj) : GraphicsAPIObjLink()
				{
					obj = std::move(_obj);
				}
				GraphicsAPIObjLink(GraphicsAPIObjLink&& other) : GraphicsAPIObjLink()
				{
					operator=(std::move(other));
				}

				GraphicsAPIObjLink(const GraphicsAPIObjLink& other) = delete;
				GraphicsAPIObjLink& operator=(const GraphicsAPIObjLink& other) = delete;
				GraphicsAPIObjLink& operator=(GraphicsAPIObjLink&& other)
				{
					std::swap(obj,other.obj);
					std::swap(cudaHandle,other.cudaHandle);
					std::swap(acquired,other.acquired);
					std::swap(asImage,other.asImage);
					return *this;
				}

				~GraphicsAPIObjLink()
				{
					assert(!acquired); // you've fucked up, there's no way for us to fix it, you need to release the objects on a proper stream
					if (obj)
						CCUDAHandler::cuda.pcuGraphicsUnregisterResource(cudaHandle);
				}

				//
				auto* getObject() const {return obj.get();}

			private:
				core::smart_refctd_ptr<ObjType> obj;
				CUgraphicsResource cudaHandle;
				bool acquired;

				friend class CCUDAHandler;
			public:
				union
				{
					struct
					{
						CUdeviceptr pointer;
					} asBuffer;
					struct
					{
						CUmipmappedArray mipmappedArray;
						CUarray array;
					} asImage;
				};
		};

		//
		static CUresult registerBuffer(GraphicsAPIObjLink<video::IGPUBuffer>* link, uint32_t flags = CU_GRAPHICS_REGISTER_FLAGS_NONE);
		static CUresult registerImage(GraphicsAPIObjLink<video::IGPUImage>* link, uint32_t flags = CU_GRAPHICS_REGISTER_FLAGS_NONE);
		

		template<typename ObjType>
		static CUresult acquireResourcesFromGraphics(void* tmpStorage, GraphicsAPIObjLink<ObjType>* linksBegin, GraphicsAPIObjLink<ObjType>* linksEnd, CUstream stream)
		{
			auto count = std::distance(linksBegin,linksEnd);

			auto resources = reinterpret_cast<CUgraphicsResource*>(tmpStorage);
			auto rit = resources;
			for (auto iit=linksBegin; iit!=linksEnd; iit++,rit++)
			{
				if (iit->acquired)
					return CUDA_ERROR_UNKNOWN;
				*rit = iit->cudaHandle;
			}

			auto retval = cuda.pcuGraphicsMapResources(count,resources,stream);
			for (auto iit=linksBegin; iit!=linksEnd; iit++)
				iit->acquired = true;
			return retval;
		}
		template<typename ObjType>
		static CUresult releaseResourcesToGraphics(void* tmpStorage, GraphicsAPIObjLink<ObjType>* linksBegin, GraphicsAPIObjLink<ObjType>* linksEnd, CUstream stream)
		{
			auto count = std::distance(linksBegin,linksEnd);

			auto resources = reinterpret_cast<CUgraphicsResource*>(tmpStorage);
			auto rit = resources;
			for (auto iit=linksBegin; iit!=linksEnd; iit++,rit++)
			{
				if (!iit->acquired)
					return CUDA_ERROR_UNKNOWN;
				*rit = iit->cudaHandle;
			}

			auto retval = cuda.pcuGraphicsUnmapResources(count,resources,stream);
			for (auto iit=linksBegin; iit!=linksEnd; iit++)
				iit->acquired = false;
			return retval;
		}

		static CUresult acquireAndGetPointers(GraphicsAPIObjLink<video::IGPUBuffer>* linksBegin, GraphicsAPIObjLink<video::IGPUBuffer>* linksEnd, CUstream stream, size_t* outbufferSizes = nullptr);
		static CUresult acquireAndGetMipmappedArray(GraphicsAPIObjLink<video::IGPUImage>* linksBegin, GraphicsAPIObjLink<video::IGPUImage>* linksEnd, CUstream stream);
		static CUresult acquireAndGetArray(GraphicsAPIObjLink<video::IGPUImage>* linksBegin, GraphicsAPIObjLink<video::IGPUImage>* linksEnd, uint32_t* arrayIndices, uint32_t* mipLevels, CUstream stream);
#endif
		CUdevice getInternalObject() const { return m_handle; }
		const CCUDAHandler* getHandler() const { return m_handler.get();  }
		core::smart_refctd_ptr<IGPUBuffer> exportGPUBuffer(CCUDASharedMemory* mem, ILogicalDevice* device);
		CUresult importGPUBuffer(core::smart_refctd_ptr<CCUDASharedMemory>* outPtr, IGPUBuffer* buf);
		CUresult importGPUSemaphore(core::smart_refctd_ptr<CCUDASharedSemaphore>* outPtr, IGPUSemaphore* sem);
		CUresult createExportableMemory(core::smart_refctd_ptr<CCUDASharedMemory>* outMem, size_t size, size_t alignment);
		
	protected:
		friend class CCUDAHandler;
		friend class CCUDASharedMemory;
		friend class CCUDASharedSemaphore;

		struct SCUDACleaner : video::ICleanup
		{
			core::smart_refctd_ptr<core::IReferenceCounted> resource;
			SCUDACleaner(core::smart_refctd_ptr<core::IReferenceCounted> resource)
				: resource(std::move(resource))
			{ }
			~SCUDACleaner() override
			{
				resource = nullptr;
			}
		};

		CUresult reserveAdrressAndMapMemory(CUdeviceptr* outPtr, size_t size, size_t alignment, CUmemGenericAllocationHandle memory);
		
		CCUDADevice(core::smart_refctd_ptr<CVulkanConnection>&& _vulkanConnection, IPhysicalDevice* const _vulkanDevice, const E_VIRTUAL_ARCHITECTURE _virtualArchitecture, CUdevice _handle, core::smart_refctd_ptr<CCUDAHandler>&& _handler);
		~CCUDADevice();
		
		std::vector<const char*> m_defaultCompileOptions;
		core::smart_refctd_ptr<CVulkanConnection> m_vulkanConnection;
		IPhysicalDevice* const m_vulkanDevice;
		E_VIRTUAL_ARCHITECTURE m_virtualArchitecture;
		core::smart_refctd_ptr<CCUDAHandler> m_handler;
		CUdevice m_handle;
		CUcontext m_context;
};

}

#endif // _NBL_COMPILE_WITH_CUDA_

#endif
