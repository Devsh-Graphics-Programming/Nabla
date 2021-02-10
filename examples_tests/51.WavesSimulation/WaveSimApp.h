#pragma once
#include <nabla.h>

struct WaveSimParams
{
	//Both width and height MUST be powers of 2
	union
	{
		struct
		{
			uint32_t width, height;
		};
		nbl::core::dimension2du size;
	};

};

class WaveSimApp
{
	using computePipeline = nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline>;
	using graphicsPipeline = nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>;
	using textureView = nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView>;
private:
	[[nodiscard]] bool Init();
	[[nodiscard]] bool CreatePresentingPipeline();
	textureView CreateTexture(nbl::core::dimension2du size, nbl::asset::E_FORMAT format = nbl::asset::E_FORMAT::EF_R8G8B8A8_UNORM) const;
	void PresentWaves(textureView tex);
public:
	WaveSimApp(const WaveSimParams& params);
	void Run();
private:
	WaveSimParams m_params;
	
private:
	graphicsPipeline m_presenting_pipeline;
	nbl::core::smart_refctd_ptr<nbl::IrrlichtDevice> m_device;
	nbl::video::IVideoDriver* m_driver;
	nbl::io::IFileSystem* m_filesystem;
	nbl::asset::IAssetManager* m_asset_manager;

	nbl::core::smart_refctd_ptr<nbl::video::IGPUMeshBuffer> m_current_gpu_mesh_buffer;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_gpu_descriptor_set_layout;
};