#ifndef _RENDERER_INCLUDED_
#define _RENDERER_INCLUDED_

#include "nabla.h"
#include "nbl/ext/RadeonRays/RadeonRays.h"
// pesky leaking defines
#undef PI

#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"

#include <ISceneManager.h>

#ifdef _NBL_BUILD_OPTIX_
#include "nbl/ext/OptiX/Manager.h"
#endif

#include <thread>
#include <future>
#include <filesystem>

class Renderer : public nbl::core::IReferenceCounted, public nbl::core::InterfaceUnmovable
{
    public:
		#include "rasterizationCommon.h"
		#include "raytraceCommon.h"
		#ifdef __cplusplus
			#undef uint
			#undef vec4
			#undef mat4
			#undef mat4x3
		#endif
		
		struct DenoiserArgs
		{
			std::filesystem::path bloomFilePath;
			float bloomScale = 0.0f;
			float bloomIntensity = 0.0f;
			std::string tonemapperArgs = "";
		};

		Renderer(nbl::video::IVideoDriver* _driver, nbl::asset::IAssetManager* _assetManager, nbl::scene::ISceneManager* _smgr, bool useDenoiser = true);

		void initSceneResources(nbl::asset::SAssetBundle& meshes, nbl::io::path&& _sampleSequenceCachePath="");

		void deinitSceneResources();
		
		void initScreenSizedResources(uint32_t width, uint32_t height);

		void deinitScreenSizedResources();

		void resetSampleAndFrameCounters();

		void takeAndSaveScreenShot(const std::filesystem::path& screenshotFilePath, bool denoise = false, const DenoiserArgs& denoiserArgs = {});
		
		void denoiseCubemapFaces(std::filesystem::path filePaths[6], const std::string& mergedFileName, int borderPixels, const DenoiserArgs& denoiserArgs = {});

		bool render(nbl::ITimer* timer, const bool transformNormals, const bool beauty=true);

		auto* getColorBuffer() { return m_colorBuffer; }

		const auto& getSceneBound() const { return m_sceneBound; }
		
		uint64_t getSamplesPerPixelPerDispatch() const
		{
			return m_staticViewData.samplesPerPixelPerDispatch;
		}
		uint64_t getTotalSamplesPerPixelComputed() const
		{
			const auto framesDispatched = static_cast<uint64_t>(m_framesDispatched);
			return framesDispatched*getSamplesPerPixelPerDispatch();
		}
		uint64_t getTotalSamplesComputed() const
		{
			const auto samplesPerDispatch = getSamplesPerPixelPerDispatch()*static_cast<uint64_t>(m_staticViewData.imageDimensions.x*m_staticViewData.imageDimensions.y);
			const auto framesDispatched = static_cast<uint64_t>(m_framesDispatched);
			return framesDispatched*samplesPerDispatch;
		}
		uint64_t getTotalRaysCast() const
		{
			return m_totalRaysCast;
		}

		//! Brief guideline to good path depth limits
		// Want to see stuff with indirect lighting on the other side of a pane of glass
		// 5 = glass frontface->glass backface->diffuse surface->diffuse surface->light
		// Want to see through a glass box, vase, or office 
		// 7 = glass frontface->glass backface->glass frontface->glass backface->diffuse surface->diffuse surface->light
		// pick higher numbers for better GI and less bias
		static inline constexpr uint32_t DefaultPathDepth = 8u;
		// TODO: Upload only a subsection of the sample sequence to the GPU, so we can use more samples without trashing VRAM
		static inline constexpr uint32_t MaxFreeviewSamples = 0x10000u;

		//
		static constexpr inline uint32_t AntiAliasingSequenceLength = 1024;
		static const float AntiAliasingSequence[AntiAliasingSequenceLength][2];
    protected:
        ~Renderer();

		struct InitializationData
		{
			InitializationData() : lights(),lightCDF() {}
			InitializationData(InitializationData&& other) : InitializationData()
			{
				operator=(std::move(other));
			}
			~InitializationData() {lightCDF.~vector(); }

			inline InitializationData& operator=(InitializationData&& other)
			{
				lights = std::move(other.lights);
				lightCDF = std::move(other.lightCDF);
				return *this;
			}

			nbl::core::vector<SLight> lights;
			union
			{
				nbl::core::vector<float> lightPDF;
				nbl::core::vector<uint32_t> lightCDF;
			};
		};
		InitializationData initSceneObjects(const nbl::asset::SAssetBundle& meshes);
		void initSceneNonAreaLights(InitializationData& initData);
		void finalizeScene(InitializationData& initData);

		//
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> createScreenSizedTexture(nbl::asset::E_FORMAT format, uint32_t layers=0u);

		//
		void preDispatch(const nbl::video::IGPUPipelineLayout* layout, nbl::video::IGPUDescriptorSet*const *const lastDS);
		bool traceBounce(uint32_t& inoutRayCount);

		//
		const nbl::ext::MitsubaLoader::CMitsubaMetadata* m_globalMeta = nullptr;

		// "constants"
		bool m_useDenoiser;

		// managers
		nbl::video::IVideoDriver* m_driver;

		nbl::asset::IAssetManager* m_assetManager;
		nbl::scene::ISceneManager* m_smgr;

		nbl::core::smart_refctd_ptr<nbl::ext::RadeonRays::Manager> m_rrManager;


		// persistent (intialized in constructor
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_rayCountBuffer,m_littleDownloadBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_cullDSLayout;
		nbl::core::smart_refctd_ptr<const nbl::video::IGPUDescriptorSetLayout> m_perCameraRasterDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_rasterInstanceDataDSLayout,m_additionalGlobalDSLayout,m_commonRaytracingDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_raygenDSLayout,m_closestHitDSLayout,m_resolveDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline> m_visibilityBufferFillPipeline;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_cullPipelineLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_raygenPipelineLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_closestHitPipelineLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_resolvePipelineLayout;
		
		nbl::core::smart_refctd_ptr<IGPUSpecializedShader> m_cullGPUShader;
		nbl::core::smart_refctd_ptr<IGPUSpecializedShader> m_raygenGPUShader;
		nbl::core::smart_refctd_ptr<IGPUSpecializedShader> m_closestHitGPUShader;
		nbl::core::smart_refctd_ptr<IGPUSpecializedShader> m_resolveGPUShader;

		// semi persistent data
		nbl::io::path sampleSequenceCachePath;
		struct SampleSequence
		{
			public:
				static inline constexpr auto QuantizedDimensionsBytesize = sizeof(uint64_t);
				SampleSequence() : bufferView() {}

				// one less because first path vertex uses a different sequence 
				static inline uint32_t computeQuantizedDimensions(uint32_t maxPathDepth) {return (maxPathDepth-1)*SAMPLING_STRATEGY_COUNT;}
				nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer> createCPUBuffer(uint32_t quantizedDimensions, uint32_t sampleCount);

				// from cache
				void createBufferView(nbl::video::IVideoDriver* driver, nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer>&& buff);
				// regenerate
				nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer> createBufferView(nbl::video::IVideoDriver* driver, uint32_t quantizedDimensions, uint32_t sampleCount);

				auto getBufferView() const {return bufferView;}

			private:
				nbl::core::smart_refctd_ptr<nbl::video::IGPUBufferView> bufferView;
		} sampleSequence;
		uint16_t pathDepth;
		uint16_t noRussianRouletteDepth;
		uint32_t maxSensorSamples;

		// scene specific data
		nbl::core::vector<::RadeonRays::Shape*> rrShapes;
		nbl::core::vector<::RadeonRays::Shape*> rrInstances;

		nbl::core::matrix3x4SIMD m_prevView;
		nbl::core::matrix4x3 m_prevCamTform;
		nbl::core::aabbox3df m_sceneBound;
		uint32_t m_framesDispatched;
		vec2 m_rcpPixelSize;
		uint64_t m_totalRaysCast;
		StaticViewData_t m_staticViewData;
		RaytraceShaderCommonData_t m_raytraceCommonData;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_indexBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_indirectDrawBuffers[2];
		struct MDICall
		{
			uint32_t mdiOffset,mdiCount;
		};
		nbl::core::vector<MDICall> m_mdiDrawCalls;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_cullDS;
		CullShaderData_t m_cullPushConstants;
		uint32_t m_cullWorkGroups;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_perCameraRasterDS;
		
		nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_cullPipeline,m_raygenPipeline,m_closestHitPipeline,m_resolvePipeline;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_globalBackendDataDS,m_additionalGlobalDS;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_commonRaytracingDS[2];
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_rasterInstanceDataDS,m_raygenDS,m_resolveDS;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_closestHitDS[2];
		uint32_t m_raygenWorkGroups[2];

		struct InteropBuffer
		{
			nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> buffer;
			std::pair<::RadeonRays::Buffer*, cl_mem> asRRBuffer = { nullptr,0u };
		};
		InteropBuffer m_rayBuffer[2];
		InteropBuffer m_intersectionBuffer[2];
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_accumulation,m_tonemapOutput;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_albedoAcc,m_albedoRslv;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_normalAcc,m_normalRslv;
		nbl::video::IFrameBuffer* m_visibilityBuffer,* m_colorBuffer;
		
		// Resources used for blending environmental maps
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline> blendEnvPipeline;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> blendEnvDescriptorSet;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUMeshBuffer> blendEnvMeshBuffer;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_finalEnvmap;

		std::future<bool> compileShadersFuture;
};

#endif
