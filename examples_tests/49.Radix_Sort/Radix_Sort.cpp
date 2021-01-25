#include "Radix_Sort.h"

void Radix_Sort::Radix_Sort::CreateFilledLocalBuffer() noexcept
{
	GPU_Buffer = (*Video_Driver.get())->createFilledDeviceLocalGPUBufferOnDedMem(Radix_Array_Ptr->Get_Data_Size(), static_cast<void*>(Radix_Array_Ptr->begin()));
}

void Radix_Sort::Radix_Sort::Init()
{
	CreateFilledLocalBuffer();

	nbl::io::IFileSystem* filesystem = Irrlicht_Device->getFileSystem();
	nbl::video::IVideoDriver* Video_Driver_Ptr = (*Video_Driver.get());
	
	auto f = nbl::core::smart_refctd_ptr<nbl::io::IReadFile>(filesystem->createAndOpenFile("../compute_shader.comp"));
	
	Video_Driver_Ptr->bindComputePipeline(compPipeline.get());
}
