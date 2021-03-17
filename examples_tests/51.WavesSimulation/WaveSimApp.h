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
	nbl::core::vector2df length_unit;
	nbl::core::vector2df wind_dir;
	float wind_speed;
	float amplitude; 
	float wind_dependency;
};

class WaveSimApp
{
	struct MeshData
	{
		nbl::asset::SVertexInputParams input_params;
		uint32_t index_count;
	};
	using computePipeline = nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline>;
	using graphicsPipeline = nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>;
	using textureView = nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView>;
private:
	[[nodiscard]] bool Init();
	[[nodiscard]] bool CreatePresentingPipeline();
	[[nodiscard]] bool CreateComputePipelines();
	textureView CreateTexture(nbl::core::dimension2du size, nbl::asset::E_FORMAT format = nbl::asset::E_FORMAT::EF_R8G8B8A8_UNORM) const;
	void PresentWaves2D(const textureView& tex);
	void PresentWaves3D(const textureView& tex);
	nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> RandomizeWaveSpectrum();
	void AnimateSpectrum(const nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>& h0, nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>& animated_spectrum, float time);
	void GenerateHeightMap(const nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>& h0, textureView& out, float time);
	void GenerateNormalMap(const textureView& heightmap, textureView& normalmap);
	MeshData CreateRectangularWavesMesh();
public:
	WaveSimApp(const WaveSimParams& params);
	void Run();
private:
	WaveSimParams m_params;
	
private:
	nbl::core::smart_refctd_ptr<nbl::IrrlichtDevice> m_device;
	nbl::video::IVideoDriver* m_driver;
	nbl::io::IFileSystem* m_filesystem;
	nbl::asset::IAssetManager* m_asset_manager;

	graphicsPipeline m_presenting_pipeline;
	computePipeline m_spectrum_randomizing_pipeline;
	computePipeline m_spectrum_animating_pipeline;
	computePipeline m_ifft_pipeline_1;
	computePipeline m_ifft_pipeline_2;
	computePipeline m_normalmap_generating_pipeline;

	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_randomizer_descriptor_set;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_spectrum_animating_descriptor_set;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_ifft_1_descriptor_set;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_ifft_2_descriptor_set;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_normalmap_descriptor_set;

	nbl::core::smart_refctd_ptr<nbl::video::IGPUMeshBuffer> m_current_gpu_mesh_buffer;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_gpu_descriptor_set_layout;
};