#ifndef _EXTRA_CRAP_INCLUDED_
#define _EXTRA_CRAP_INCLUDED_

#include "nabla.h"

#include "nbl/ext/RadeonRays/RadeonRays.h"
// pesky leaking defines
#undef PI

#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"

#include <ISceneManager.h>

#ifdef _NBL_BUILD_OPTIX_
#include "nbl/ext/OptiX/Manager.h"
#endif


class Renderer : public nbl::core::IReferenceCounted, public nbl::core::InterfaceUnmovable
{
    public:
		#include "../drawCommon.glsl"
		#ifdef __cplusplus
			#undef mat4
			#undef mat4x3
		#endif

		// No 8k yet, too many rays to store
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxResolution[2] = {7680/2,4320/2};


		Renderer(nbl::video::IVideoDriver* _driver, nbl::asset::IAssetManager* _assetManager, nbl::scene::ISceneManager* _smgr, nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet>&& globalBackendDataDS, bool useDenoiser = true);

		void init(	const nbl::asset::SAssetBundle& meshes,
					bool isCameraRightHanded,
					nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer>&& sampleSequence,
					uint32_t rayBufferSize=(sizeof(::RadeonRays::ray)*2u+sizeof(uint32_t)*2u)*MaxResolution[0]*MaxResolution[1]); // 2 samples for MIS, TODO: compute default buffer size

		void deinit();

		void render(nbl::ITimer* timer);

		bool isRightHanded() { return m_rightHanded; }

		auto* getColorBuffer() { return m_colorBuffer; }

		const auto& getSceneBound() const { return m_sceneBound; }

		uint64_t getTotalSamplesComputed() const { return static_cast<uint64_t>(m_samplesComputedPerPixel)*static_cast<uint64_t>(m_rayCountPerDispatch)/m_samplesPerPixelPerDispatch; }


		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxDimensions = 6u;
    protected:
        ~Renderer();

		struct InitializationData
		{
			InitializationData() : lights(),lightRadiances(),lightCDF(),globalMeta(nullptr) {}
			InitializationData(InitializationData&& other) : InitializationData()
			{
				operator=(std::move(other));
			}
			~InitializationData() {lightCDF.~vector(); }

			inline InitializationData& operator=(InitializationData&& other)
			{
				lights = std::move(other.lights);
				lightRadiances = std::move(other.lightRadiances);
				lightCDF = std::move(other.lightCDF);
				globalMeta = other.globalMeta;
				return *this;
			}


			nbl::core::vector<SLight> lights;
			nbl::core::vector<nbl::core::vectorSIMDf> lightRadiances;
			union
			{
				nbl::core::vector<float> lightPDF;
				nbl::core::vector<uint32_t> lightCDF;
			};
			const nbl::ext::MitsubaLoader::CGlobalMitsubaMetadata* globalMeta = nullptr;
		};
		InitializationData initSceneObjects(const nbl::asset::SAssetBundle& meshes);
		void initSceneNonAreaLights(InitializationData& initData);
		void finalizeScene(InitializationData& initData);

		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> createScreenSizedTexture(nbl::asset::E_FORMAT format);

#if TODO
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDS2layoutCompost(bool useDenoiser, core::smart_refctd_ptr<video::IGPUSampler>& nearstSampler);
		core::smart_refctd_ptr<video::IGPUDescriptorSet> createDS2Compost(bool useDenoiser,
			core::smart_refctd_ptr<video::IGPUSampler>& nearestSampler
		);
		core::smart_refctd_ptr<video::IGPUPipelineLayout> createLayoutCompost();

		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDS2layoutRaygen(core::smart_refctd_ptr<video::IGPUSampler>& nearstSampler);
		core::smart_refctd_ptr<video::IGPUDescriptorSet> createDS2Raygen(core::smart_refctd_ptr<video::IGPUSampler>& nearstSampler);
		core::smart_refctd_ptr<video::IGPUPipelineLayout> createLayoutRaygen();
#endif

		const bool m_useDenoiser;

        nbl::video::IVideoDriver* m_driver;

		nbl::asset::IAssetManager* m_assetManager;
		nbl::scene::ISceneManager* m_smgr;

		nbl::core::smart_refctd_ptr<nbl::ext::RadeonRays::Manager> m_rrManager;

		nbl::core::smart_refctd_ptr<nbl::asset::ICPUSpecializedShader> m_visibilityBufferFillShaders[2];
		nbl::core::smart_refctd_ptr<nbl::asset::ICPUPipelineLayout> m_visibilityBufferFillPipelineLayoutCPU;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_visibilityBufferFillPipelineLayoutGPU;
		nbl::core::smart_refctd_ptr<const nbl::video::IGPUDescriptorSetLayout> m_perCameraRasterDSLayout;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_cullDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_cullPipelineLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_cullPipeline;
		
		nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_raygenPipeline,m_resolvePipeline;


		nbl::core::vectorSIMDf baseEnvColor;
		nbl::core::aabbox3df m_sceneBound;
		uint32_t m_renderSize[2u];
		bool m_rightHanded;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_indirectDrawBuffers[2];
		struct MDICall
		{
			nbl::asset::SBufferBinding<nbl::video::IGPUBuffer> vertexBindings[nbl::video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
			nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> indexBuffer;
			nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline> pipeline;
			uint32_t mdiOffset,mdiCount;
		};
		nbl::core::vector<MDICall> m_mdiDrawCalls;
		CullShaderData_t m_cullPushConstants;

		uint32_t m_lightCount;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_lightCDFBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_lightBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_lightRadianceBuffer;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_globalBackendDataDS,m_cullDS,m_perCameraRasterDS; // TODO: do we need to keep track of this?


		nbl::ext::RadeonRays::Manager::MeshBufferRRShapeCache rrShapeCache;
#if TODO
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_compostDS2;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_compostLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_compostPipeline;

		uint32_t m_raygenWorkGroups[2];
		uint32_t m_resolveWorkGroups[2];

		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_raygenDS2;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_raygenLayout;

		nbl::ext::RadeonRays::Manager::MeshNodeRRInstanceCache rrInstances;
#endif
		struct InteropBuffer
		{
			nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> buffer;
			std::pair<::RadeonRays::Buffer*,cl_mem> asRRBuffer;
		};
		InteropBuffer m_rayCountBuffer,m_rayBuffer,m_intersectionBuffer;


		enum E_VISIBILITY_BUFFER_ATTACHMENT
		{
			EVBA_DEPTH,
			EVBA_OBJECTID_AND_TRIANGLEID_AND_FRONTFACING,
			// TODO: Once we get geometry packer V2 (virtual geometry) no need for these buffers actually (might want/need a barycentric buffer)
			EVBA_NORMALS,
			EVBA_UV_COORDINATES,
			EVBA_COUNT
		};
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_visibilityBufferAttachments[EVBA_COUNT];

		uint32_t m_maxSamples, m_samplesPerPixelPerDispatch, m_rayCountPerDispatch;
		uint32_t m_framesDone, m_samplesComputedPerPixel;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBufferView> m_sampleSequence;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_scrambleTexture;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_accumulation, m_tonemapOutput;
		nbl::video::IFrameBuffer* m_visibilityBuffer,* m_colorBuffer,* tmpTonemapBuffer;

	#ifdef _NBL_BUILD_OPTIX_
		nbl::core::smart_refctd_ptr<nbl::ext::OptiX::Manager> m_optixManager;
		CUstream m_cudaStream;
		nbl::core::smart_refctd_ptr<nbl::ext::OptiX::IContext> m_optixContext;
		nbl::core::smart_refctd_ptr<nbl::ext::OptiX::IDenoiser> m_denoiser;
		OptixDenoiserSizes m_denoiserMemReqs;
		nbl::cuda::CCUDAHandler::GraphicsAPIObjLink<nbl::video::IGPUBuffer> m_denoiserInputBuffer,m_denoiserStateBuffer,m_denoisedBuffer,m_denoiserScratchBuffer;

		enum E_DENOISER_INPUT
		{
			EDI_COLOR,
			EDI_ALBEDO,
			EDI_NORMAL,
			EDI_COUNT
		};
		OptixImage2D m_denoiserOutput;
		OptixImage2D m_denoiserInputs[EDI_COUNT];
	#endif
};

#endif
