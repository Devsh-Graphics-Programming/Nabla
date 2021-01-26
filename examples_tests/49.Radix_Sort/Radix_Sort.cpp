#include "Radix_Sort.h"

void Radix_Sort::Radix_Sort::CreateFilledLocalBuffer() noexcept
{
	GPU_Buffer = Video_Driver->createFilledDeviceLocalGPUBufferOnDedMem(Radix_Array_Ptr->Get_Data_Size(), static_cast<void*>(Radix_Array_Ptr->begin()));
}

void Radix_Sort::Radix_Sort::Init()
{
	CreateFilledLocalBuffer();

	nbl::io::IFileSystem* filesystem = Irrlicht_Device->getFileSystem();
	
	auto f = nbl::core::smart_refctd_ptr<nbl::io::IReadFile>(filesystem->createAndOpenFile("../compute_shader.comp"));
	
	
}

void Radix_Sort::Radix_Sort::Execute()
{
	Video_Driver->bindComputePipeline(compPipeline.get());
	if(Video_Driver->dispatch(1024,1024,0))
	{
		nbl::video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_ALL_BARRIER_BITS);

		//nbl::video::COpenGLExtensionHandler::extGlDeleteShader(cs);
	}
}
