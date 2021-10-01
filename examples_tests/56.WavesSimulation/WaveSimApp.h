#pragma once
#include <nabla.h>

#include "../common/QToQuitEventReceiver.h"

using namespace nbl;
using namespace asset;
using namespace video;
using namespace core;

enum class PresentingMode
{
	PM_2D,
	PM_3D
};

constexpr PresentingMode CURRENT_PRESENTING_MODE = PresentingMode::PM_3D;

struct WaveSimParams
{
	//Both width and height MUST be powers of 2
	union
	{
		struct
		{
			uint32_t width, length;
		};
		dimension2du size;
	};
	vector2df length_unit;
	vector2df wind_dir;
	float wind_speed;
	float amplitude; 
	float wind_dependency;
	float choppiness;
};


class WaveSimApp
{
	using computePipeline = smart_refctd_ptr<IGPUComputePipeline>;
	using graphicsPipeline = smart_refctd_ptr<IGPURenderpassIndependentPipeline>;
	using textureView = smart_refctd_ptr<IGPUImageView>;
private:
	[[nodiscard]] bool Init();
	[[nodiscard]] bool CreatePresenting2DPipeline();
	[[nodiscard]] bool CreatePresenting3DPipeline();
	[[nodiscard]] bool CreateSkyboxPresentingPipeline();
	[[nodiscard]] bool CreateComputePipelines();
	textureView CreateTexture(dimension2du size, E_FORMAT format = E_FORMAT::EF_R8G8B8A8_UNORM) const;
	textureView CreateTextureFromImageFile(const std::string_view image_file_path, E_FORMAT format = E_FORMAT::EF_UNKNOWN) const;
	void PresentWaves2D(const textureView& tex);
	void PresentSkybox(const textureView& envmap, matrix4SIMD mvp);
	void PresentWaves3D(const textureView& displacement_map, const textureView& normal_map, const textureView& env_map, const matrix4SIMD& mvp, const vector3df& camera);
	void GenerateDisplacementMap(const smart_refctd_ptr<IGPUBuffer>& h0, textureView& out, float time);
	void GenerateNormalMap(const textureView& displacement_map, textureView& normalmap);
	smart_refctd_ptr<IGPUBuffer> GenerateWaveSpectrum();

	smart_refctd_ptr<IGPUSpecializedShader> createGPUSpecializedShaderFromFile(const std::string_view filepath, asset::ISpecializedShader::E_SHADER_STAGE stage);
	smart_refctd_ptr<IGPUSpecializedShader> createGPUSpecializedShaderFromFileWithIncludes(const std::string_view filepath, asset::ISpecializedShader::E_SHADER_STAGE stage, std::string_view origFilePath);
public:
	WaveSimApp(const WaveSimParams& params);
	void Run();
private:
	const std::string m_envmap_file_path = "../../media/envmap/envmap_1.exr";
private:
	WaveSimParams m_params;
	
private:
	smart_refctd_ptr<IrrlichtDevice> m_device;
	IVideoDriver* m_driver;
	io::IFileSystem* m_filesystem;
	IAssetManager* m_asset_manager;

	graphicsPipeline m_presenting_pipeline;
	graphicsPipeline m_skybox_pipeline;
	computePipeline m_spectrum_randomizing_pipeline;
	computePipeline m_ifft_pipeline_1;
	computePipeline m_ifft_pipeline_2;
	computePipeline m_normalmap_generating_pipeline;

	smart_refctd_ptr<IGPUDescriptorSet> m_randomizer_descriptor_set;
	smart_refctd_ptr<IGPUDescriptorSet> m_ifft_1_descriptor_set;
	smart_refctd_ptr<IGPUDescriptorSet> m_ifft_2_descriptor_set;
	smart_refctd_ptr<IGPUDescriptorSet> m_normalmap_descriptor_set;
	smart_refctd_ptr<IGPUDescriptorSet> m_3d_presenting_descriptor_set;
	smart_refctd_ptr<IGPUDescriptorSet> m_skybox_presenting_descriptor_set;

	smart_refctd_ptr<IGPUMeshBuffer> m_2d_mesh_buffer;
	smart_refctd_ptr<IGPUMeshBuffer> m_3d_mesh_buffer;
	smart_refctd_ptr<IGPUMeshBuffer> m_skybox_mesh_buffer;

	smart_refctd_ptr<IGPUDescriptorSetLayout> m_gpu_descriptor_set_layout_2d;
	smart_refctd_ptr<IGPUDescriptorSetLayout> m_gpu_descriptor_set_layout_skybox;
	
	core::smart_refctd_ptr<video::IGPUMeshBuffer> m_gpu_sphere;

	QToQuitEventReceiver m_receiver;
};