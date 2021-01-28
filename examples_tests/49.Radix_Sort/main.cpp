#include <memory>


//#define _IRR_STATIC_LIB_
#include "Radix_Sort.h"	//my header


constexpr std::size_t Buffer_Size = 8u * 1024u * 1024u;


int main()
{
	//Using space
	using Radix_Sort::Radix_Sort;
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

	if(!device)
	{
		return 1;
	}

	
	nbl::video::IVideoDriver* Video_driver = device->getVideoDriver();
	std::unique_ptr<Radix_Sort::Sorter> Radix_Sort_Ptr = std::make_unique<Radix_Sort>(Video_driver,device, Buffer_Size);
	Radix_Sort_Ptr->Init();

	return 0;
}