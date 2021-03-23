#pragma once
#include <nabla.h>

#include "../common/QToQuitEventReceiver.h"

struct WaveSimParams
{
	//Both width and height MUST be powers of 2
	union
	{
		struct
		{
			uint32_t width, length;
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
	using computePipeline = nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline>;
	using graphicsPipeline = nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>;
	using textureView = nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView>;
private:
	[[nodiscard]] bool Init();
	[[nodiscard]] bool CreatePresenting2DPipeline();
	[[nodiscard]] bool CreatePresenting3DPipeline();
	[[nodiscard]] bool CreateComputePipelines();
	textureView CreateTexture(nbl::core::dimension2du size, nbl::asset::E_FORMAT format = nbl::asset::E_FORMAT::EF_R8G8B8A8_UNORM) const;
	void PresentWaves2D(const textureView& tex);
	void PresentWaves3D(const textureView& displacement_map, const textureView& normal_map, const nbl::core::matrix4SIMD& mvp, const nbl::core::vector3df& camera);
	nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> RandomizeWaveSpectrum();
	void AnimateSpectrum(const nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>& h0, nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>& animated_spectrum, float time);
	void GenerateDisplacementMap(const nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>& h0, textureView& out, float time);
	void GenerateNormalMap(const textureView& displacement_map, textureView& normalmap);
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
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_presenting_3d_descriptor_set;

	nbl::core::smart_refctd_ptr<nbl::video::IGPUMeshBuffer> m_2d_mesh_buffer;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUMeshBuffer> m_3d_mesh_buffer;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_gpu_descriptor_set_layout_2d;

	QToQuitEventReceiver m_receiver;

};