#include "Radix_Sort.h"

#include <gli/image.hpp>
#include "nbl/nblpack.h"

//void Radix_Sort::Radix_Sort::CreateFilledLocalBuffer() noexcept
//{
//	GPU_Buffer = Video_Driver->createFilledDeviceLocalGPUBufferOnDedMem(Radix_Array_Ptr->Get_Data_Size(), static_cast<void*>(Radix_Array_Ptr->begin()));
//}

using buffer_type = uint16_t;
constexpr std::size_t buffer_size = 10;

struct alignas(16) SShaderStorageBufferObject //MOJE SSBO BUFFER
{

	buffer_type buffer[buffer_size]{ 1,2,3,4,5,6,7,8,9,10 };

} PACK_STRUCT;


void Fill_Buffer_Ascending_Order(SShaderStorageBufferObject* ssbo)//Fill array with ascending order
{

	auto get_random = [&](const buffer_type& min, const buffer_type& max)
	{
		static std::default_random_engine engine;
		static std::uniform_real_distribution<> distribution(min, max);
		return static_cast<buffer_type>(distribution(engine));
	};

	for (std::size_t i = 0; i < buffer_size; ++i)	
	{
		ssbo->buffer[i] = static_cast<buffer_type>(i);
	}
	
}

void Radix_Sort::Radix_Sort::Init()
{
	//CreateFilledLocalBuffer();

	//nbl::io::IFileSystem* filesystem = Irrlicht_Device->getFileSystem();
	//// auto device = createDeviceEx(params);

	auto* assetManager = Irrlicht_Device->getAssetManager();

	auto computeShaderBundle = assetManager->getAsset("../compute_shader.comp", {});	//ladowanie shadera z pliku, folderu

	assert(!computeShaderBundle.isEmpty());

	auto cpuComputeShader = nbl::core::smart_refctd_ptr_static_cast<nbl::asset::ICPUSpecializedShader>(computeShaderBundle.getContents().begin()[0]);	//wyciaganie z "paczki" moje shadera, ktorego chce
	auto gpuComputeShader = Video_Driver->getGPUObjectsFromAssets(&cpuComputeShader, &cpuComputeShader + 1)->front();											//konwertowanie cpu shadera na gpu shadera

	auto cpuSSBOBuffer = nbl::core::make_smart_refctd_ptr<nbl::asset::ICPUBuffer>(sizeof(SShaderStorageBufferObject));	//alokacja buffera mojego na gpu

	//// inicjalizuje ten buffer danymi
	Fill_Buffer_Ascending_Order(static_cast<SShaderStorageBufferObject*>(cpuSSBOBuffer->getPointer()));


	
	auto gpuSSBOOffsetBufferPair = Video_Driver->getGPUObjectsFromAssets(&cpuSSBOBuffer, &cpuSSBOBuffer + 1)->front(); //konwersja buffera z cpu na gpu
	auto gpuSSBOBuffer = nbl::core::smart_refctd_ptr<IGPUBuffer>(gpuSSBOOffsetBufferPair->getBuffer());														//wyciaganie samego buffera



	const std::size_t COUNT = 1;	//wielkosc layoutu
	nbl::video::IGPUDescriptorSetLayout::SBinding gpuBindingsLayout[COUNT] =
	{
		{0, nbl::asset::EDT_STORAGE_BUFFER, 1u, nbl::video::IGPUSpecializedShader::ESS_COMPUTE, nullptr}, //0 means, number of binding inside the shader binding = 0, etc...
	};

	auto gpuCDescriptorSetLayout = Video_Driver->createGPUDescriptorSetLayout(gpuBindingsLayout, gpuBindingsLayout + COUNT);  //uchwyt do tworzenowego obiektu, buffera, po stronie gpu. Uchwyt, zbior obiektow, jesli maja sam typ, to jeden descriptor set (max chyba 4)




	const std::size_t POSITION = 0;
	auto gpuCDescriptorSet = Video_Driver->createGPUDescriptorSet(nbl::core::smart_refctd_ptr(gpuCDescriptorSetLayout));
	{
		nbl::video::IGPUDescriptorSet::SDescriptorInfo gpuDescriptorSetInfos[COUNT];

		gpuDescriptorSetInfos[POSITION].desc = gpuSSBOBuffer;
		gpuDescriptorSetInfos[POSITION].buffer.size = sizeof(SShaderStorageBufferObject::buffer);
		gpuDescriptorSetInfos[POSITION].buffer.offset = 0; //jesli mam wiecej, to dodaje, przesuwam wskaznik o iles byte


		nbl::video::IGPUDescriptorSet::SWriteDescriptorSet gpuWrites[COUNT];
		{
			//for (uint32_t binding = 0u; binding < COUNT; binding++) //jeden binding wiec bez petli
			gpuWrites[0] = { gpuCDescriptorSet.get(), 0, 0u, 1u, nbl::asset::EDT_STORAGE_BUFFER, gpuDescriptorSetInfos + 0 };


			Video_Driver->updateDescriptorSets(COUNT, gpuWrites, 0u, nullptr);
		}
	}

	auto gpuCPipelineLayout = Video_Driver->createGPUPipelineLayout(nullptr, nullptr, std::move(gpuCDescriptorSetLayout), nullptr, nullptr, nullptr); // jak std::move, to pozycje layoutow, 0 , 1 ,2 i tak dalej
	auto gpuComputePipeline = Video_Driver->createGPUComputePipeline(nullptr, std::move(gpuCPipelineLayout), std::move(gpuComputeShader));

	///
	///
	QToQuitEventReceiver receiver;
	Irrlicht_Device->setEventReceiver(&receiver);


	uint64_t lastFPSTime{};
	while (Irrlicht_Device->run() && receiver.keepOpen())
	{
		Video_Driver->bindComputePipeline(gpuComputePipeline.get());
		//driver->pushConstants(gpuComputePipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0);
		Video_Driver->bindDescriptorSets(nbl::video::EPBP_COMPUTE, gpuComputePipeline->getLayout(), 0, 1, &gpuCDescriptorSet.get(), nullptr);

		//static_assert(NUMBER_OF_PARTICLES % WORK_GROUP_SIZE == 0, "Inccorect amount!");
		//_NBL_STATIC_INLINE_CONSTEXPR size_t groupCountX = NUMBER_OF_PARTICLES / WORK_GROUP_SIZE;

		Video_Driver->dispatch(10, 1, 1);
		nbl::video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		nbl::video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);


		uint64_t time = Irrlicht_Device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Compute Shaders - Nabla Engine [" << Video_Driver->getName() << "] FPS:" << Video_Driver->getFPS() << " PrimitvesDrawn:" << Video_Driver->getPrimitiveCountDrawn();

			Irrlicht_Device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}


	//Zmapuj memory
	{
		auto downloadStagingArea = Video_Driver->getDefaultDownStreamingBuffer();
		uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address; // remember without initializing the address to be allocated to invalid_address you won't get an allocation!
		const uint32_t Buffer_Alignment = std::alignment_of<SShaderStorageBufferObject>(); // common page size
		const uint32_t Buffer_Size = sizeof(SShaderStorageBufferObject); // common page size
	// get the data from the GPU
		{
			constexpr uint64_t timeoutInNanoSeconds = 300000000000u;
			const auto waitPoint = std::chrono::high_resolution_clock::now() + std::chrono::nanoseconds(timeoutInNanoSeconds);

			// download buffer
			{
				auto unallocatedSize = downloadStagingArea->multi_alloc(waitPoint, 1u, &address, &Buffer_Size, &Buffer_Alignment);
				if (unallocatedSize)
				{
					nbl::os::Printer::log("Could not download the buffer from the GPU!", nbl::ELL_ERROR);
					return;
				}

				Video_Driver->copyBuffer(gpuSSBOBuffer.get(), downloadStagingArea->getBuffer(), 0u, address, Buffer_Size);
			}
			auto downloadFence = Video_Driver->placeFence(true);

			//// set up regions
			//auto regions = nbl::core::make_refctd_dynamic_array<nbl::core::smart_refctd_dynamic_array<nbl::asset::IImage::SBufferCopy> >(1u);
			//{
			//	//auto& region = regions->front();
			//	//region.bufferOffset = 0u;
			//	//region.bufferRowLength = param.image[EII_COLOR]->getRegions().begin()[0].bufferRowLength;
			//	//region.bufferImageHeight = param.height;
			//	////region.imageSubresource.aspectMask = wait for Vulkan;
			//	//region.imageSubresource.mipLevel = 0u;
			//	//region.imageSubresource.baseArrayLayer = 0u;
			//	//region.imageSubresource.layerCount = 1u;
			//	//region.imageOffset = { 0u,0u,0u };
			//	//region.imageExtent = imgParams.extent;
			//}
		}
		// the cpu is not touching the data yet because the custom CPUBuffer is adopting the memory (no copy)
		auto* data = reinterpret_cast<buffer_type*>(downloadStagingArea->getBufferPointer()) + address;
		auto cpubufferalias = nbl::core::make_smart_refctd_ptr<nbl::asset::CCustomAllocatorCPUBuffer<nbl::core::null_allocator<uint8_t> > >(Buffer_Size, data, nbl::core::adopt_memory);

		auto buffer_after_compute_shader = static_cast<buffer_type*>(cpubufferalias->getPointer());
		int a = 0;
	}





	//	#include "irr/irrpack.h"
	//struct alignas(16) SShaderStorageBufferObject
	//{
	//	core::vector4df_SIMD positions[NUMBER_OF_PARTICLES];
	//	core::vector4df_SIMD velocities[NUMBER_OF_PARTICLES];
	//	core::vector4df_SIMD colors[NUMBER_OF_PARTICLES];
	//	bool isColorIntensityRising[NUMBER_OF_PARTICLES][4];
	//} PACK_STRUCT;
	//#include "irr/irrunpack.h"


}

void Radix_Sort::Radix_Sort::Execute()
{
	//Video_Driver->bindComputePipeline(compPipeline.get());
	//if(Video_Driver->dispatch(1024,1024,0))
	//{
	//	nbl::video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_ALL_BARRIER_BITS);

	//	//nbl::video::COpenGLExtensionHandler::extGlDeleteShader(cs);
	//}




}
