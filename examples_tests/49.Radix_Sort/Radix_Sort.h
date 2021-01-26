#ifndef _RADIX_SORT_H_INCLUDED_
#define _RADIX_SORT_H_INCLUDED_


#include <iostream>
#include <memory>
#include <nabla.h>
#include "Radix_Array.h"

namespace Radix_Sort
{

	class Sorter
	{
	protected:

		using IVideo_Driver = nbl::video::IVideoDriver;
		using IVideo_Driver_Ptr = nbl::video::IVideoDriver*;
		using IVideo_Driver_Ptr_Const = nbl::video::IVideoDriver* const;
		using IGPUBuffer = nbl::video::IGPUBuffer;
		using IGPUBuffer_Ptr = nbl::video::IGPUBuffer*;
		using IGPUBuffer_Const_Ptr = nbl::video::IGPUBuffer const*;
		using IGPUBuffer_Ptr_Const = nbl::video::IGPUBuffer* const;
		using IrrlichtDevice = nbl::IrrlichtDevice;

	protected:

		IVideo_Driver_Ptr Video_Driver;
		nbl::core::smart_refctd_ptr<IGPUBuffer> GPU_Buffer;
		nbl::core::smart_refctd_ptr<IrrlichtDevice> Irrlicht_Device;

	protected:

		explicit Sorter() = delete;
		explicit inline Sorter(IVideo_Driver_Ptr Video_Driver, nbl::core::smart_refctd_ptr<IrrlichtDevice> Device) :
			Video_Driver(Video_Driver),
			GPU_Buffer({}),
			Irrlicht_Device(Device)
		{}

		Sorter(const Sorter& Object) = delete;
		inline Sorter(Sorter&& Object) noexcept :
			Video_Driver(std::move(Object.Video_Driver)),
			GPU_Buffer(std::move(Object.GPU_Buffer))
		{}


		virtual void CreateFilledLocalBuffer() noexcept = 0;


		Sorter& operator=(const Sorter& Object) = delete;
		inline Sorter& operator=(Sorter&& Object) noexcept
		{
			if (this != &Object)
			{
				Video_Driver = std::move(Object.Video_Driver);
				GPU_Buffer = std::move(Object.GPU_Buffer);
			}
			return *this;
		}

	public:

		virtual void Init() = 0;
		virtual void Execute() = 0;

		virtual ~Sorter() = default;
	};




	class Radix_Sort : public Sorter
	{
		using IGPUComputePipeline = nbl::video::IGPUComputePipeline;
	private:

		nbl::core::smart_refctd_ptr<IGPUComputePipeline> compPipeline;

		//GLuint

	protected:

		std::unique_ptr<Radix_Array> Radix_Array_Ptr;

		virtual void CreateFilledLocalBuffer() noexcept override;

	public:
		Radix_Sort() = delete;
		inline Radix_Sort(IVideo_Driver_Ptr Video_Driver, nbl::core::smart_refctd_ptr<IrrlichtDevice> Device, const std::size_t Buffer_Size) :
			Sorter(Video_Driver, Device),
			Radix_Array_Ptr(std::make_unique<Radix_Array>(Buffer_Size))
		{}

		Radix_Sort(const Radix_Sort& Object) = delete;
		inline Radix_Sort(Radix_Sort&& Object) noexcept :
			Sorter(std::move(Object)),
			Radix_Array_Ptr(std::move(Object.Radix_Array_Ptr))
		{}

		virtual void Init() override;
		virtual void Execute() override;

		Radix_Sort& operator=(const Radix_Sort& Object) = delete;
		inline Radix_Sort& operator=(Radix_Sort&& Object) noexcept
		{
			if (this != &Object)
			{
				Sorter::operator=(std::move(Object));
				Radix_Array_Ptr = std::move(Object.Radix_Array_Ptr);
			}
			return *this;
		}

		~Radix_Sort() = default;
	};

}

#endif /* _RADIX_SORT_H_INCLUDED_ */