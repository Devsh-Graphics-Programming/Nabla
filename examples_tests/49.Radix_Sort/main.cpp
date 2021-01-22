#define _IRR_STATIC_LIB_
#include <nabla.h>
#include "Vec2D.h"	//my header



constexpr std::size_t Buffer_Size = 8 * 1024 * 1024;	//data size
constexpr uint32_t Alignment = sizeof(Radix_Sort::Vec2D);


int main()
{
	//Using space
	using Radix_Sort::Vec2D;
	using nbl::IrrlichtDevice;
	using nbl::SIrrlichtCreationParameters;
	using nbl::asset::SBufferRange;
	using nbl::asset::ICPUBuffer;
	using nbl::video::IGPUBuffer;
	using nbl::video::IVideoDriver;
	using nbl::core::smart_refctd_ptr;


	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24;
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = nbl::video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	params.StreamingDownloadBufferSize = Buffer_Size;


	nbl::core::smart_refctd_ptr<nbl::IrrlichtDevice> device = createDeviceEx(params);
	nbl::video::IVideoDriver* Video_driver = device->getVideoDriver();

	
	Vec2D* Data_To_Sort = new Vec2D[Buffer_Size];
	for (std::size_t i = 0; i < Buffer_Size; ++i)
	{
		Data_To_Sort[i].Set_Key(static_cast<uint32_t>(Buffer_Size - i));   //setting up the keys, Buffer_Size, Buffer_Size-1, ... ,0 and the sort it to 0, ... , Buffer_Size-1, Buffer_Size
	}

	const nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> GPU_Input_Buffer = Video_driver->createFilledDeviceLocalGPUBufferOnDedMem(Buffer_Size, Data_To_Sort);
	nbl::video::IGPUBuffer* GPU_Buffer;
	
	delete[] Data_To_Sort;	//mem free
	return 0;
}