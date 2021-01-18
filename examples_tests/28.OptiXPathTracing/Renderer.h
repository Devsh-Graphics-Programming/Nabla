// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _RENDERER_INCLUDED_
#define _RENDERER_INCLUDED_

#include "nabla.h"

#include "nbl/ext/OptiX/OptiXManager.h"


class Renderer : public nbl::core::IReferenceCounted, public nbl::core::InterfaceUnmovable
{
    public:
		struct alignas(16) SLight
		{
			enum E_TYPE : uint32_t
			{
				ET_ELLIPSOID,
				ET_TRIANGLE,
				ET_COUNT
			};
			struct CachedTransform
			{
				CachedTransform(const nbl::core::matrix3x4SIMD& tform) : transform(tform)
				{
					auto tmp4 = nbl::core::matrix4SIMD(transform.getSub3x3TransposeCofactors());
					transformCofactors = nbl::core::transpose(tmp4).extractSub3x4();
				}

				nbl::core::matrix3x4SIMD transform;
				nbl::core::matrix3x4SIMD transformCofactors;
			};


			SLight() : type(ET_COUNT) {}
			SLight(const SLight& other) : type(other.type)
			{
				std::copy(other.strengthFactor,other.strengthFactor+3u,strengthFactor);
				if (type == ET_TRIANGLE)
					std::copy(other.triangle.vertices, other.triangle.vertices+3u, triangle.vertices);
				else
				{
					analytical.transform = other.analytical.transform;
					analytical.transformCofactors = other.analytical.transformCofactors;
				}
			}
			~SLight() {}

			inline SLight& operator=(SLight&& other)
			{
				std::swap_ranges(strengthFactor,strengthFactor+3u,other.strengthFactor);
				auto a = other.analytical;
				auto t = other.triangle;
				if (type!=ET_TRIANGLE)
					other.analytical = analytical;
				else
					other.triangle = triangle;
				if (other.type)
					analytical = a;
				else
					triangle = t;
				std::swap(type, other.type);

				return *this;
			}

			void setFactor(const nbl::core::vectorSIMDf& _strengthFactor)
			{
				for (auto i=0u; i<3u; i++)
					strengthFactor[i] = _strengthFactor[i];
			}

			//! This is according to Rec.709 colorspace
			inline float getFactorLuminosity() const
			{
				float rec709LumaCoeffs[] = {0.2126f, 0.7152, 0.0722};

				//! TODO: More color spaces!
				float* colorSpaceLumaCoeffs = rec709LumaCoeffs;

				float luma = strengthFactor[0] * colorSpaceLumaCoeffs[0];
				luma += strengthFactor[1] * colorSpaceLumaCoeffs[1];
				luma += strengthFactor[2] * colorSpaceLumaCoeffs[2];
				return luma;
			}

			inline float computeAreaUnderTransform(nbl::core::vectorSIMDf differentialElementCrossProduct) const
			{
				analytical.transformCofactors.mulSub3x3WithNx1(differentialElementCrossProduct);
				return nbl::core::length(differentialElementCrossProduct).x;
			}

			inline float computeFlux(float triangulizationArea) const // also known as lumens
			{
				const auto unitHemisphereArea = 2.f*nbl::core::PI<float>();
				const auto unitSphereArea = 2.f*unitHemisphereArea;

				float lightFlux = unitHemisphereArea*getFactorLuminosity();
				switch (type)
				{
					case ET_ELLIPSOID:
						[[fallthrough]];
					case ET_TRIANGLE:
						lightFlux *= triangulizationArea;
						break;
					default:
						assert(false);
						break;
				}
				return lightFlux;
			}

			static inline SLight createFromTriangle(const nbl::core::vectorSIMDf& _strengthFactor, const CachedTransform& precompTform, const nbl::core::vectorSIMDf* v, float* outArea=nullptr)
			{
				SLight triLight;
				triLight.type = ET_TRIANGLE;
				triLight.setFactor(_strengthFactor);
				triLight.analytical = precompTform;

				float triangleArea = 0.5f*triLight.computeAreaUnderTransform(nbl::core::cross(v[1]-v[0],v[2]-v[0]));
				if (outArea)
					*outArea = triangleArea;

				for (auto k=0u; k<3u; k++)
					precompTform.transform.transformVect(triLight.triangle.vertices[k], v[k]);
				triLight.triangle.vertices[1] -= triLight.triangle.vertices[0];
				triLight.triangle.vertices[2] -= triLight.triangle.vertices[0];
				// always flip the handedness so normal points inwards (need negative normal for differential area optimization)
				if (precompTform.transform.getPseudoDeterminant().x>0.f)
					std::swap(triLight.triangle.vertices[2], triLight.triangle.vertices[1]);

				// don't do any flux magic yet

				return triLight;
			}

			//! different lights use different measures of their strength (this already has the reciprocal of the light PDF factored in)
			alignas(16) float strengthFactor[3];
			//! type is second member due to alignment issues
			E_TYPE type;
			//! useful for analytical shapes
			union
			{
				CachedTransform analytical;
				struct Triangle
				{
					nbl::core::vectorSIMDf padding[3];
					nbl::core::vectorSIMDf vertices[3];
				} triangle;
			};
			/*
			union
			{
				AreaSphere sphere;
			};
			*/
		};
		static_assert(sizeof(SLight)==112u,"Can't keep alignment straight!");

		Renderer(nbl::video::IVideoDriver* _driver, nbl::asset::IAssetManager* _assetManager, nbl::scene::ISceneManager* _smgr);

		void init(	const nbl::asset::SAssetBundle& meshes,
					bool isCameraRightHanded,
					nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer>&& sampleSequence,
					uint32_t rayBufferSize=1024u*1024u*1024u);

		void deinit();

		void render();

		bool isRightHanded() { return m_rightHanded; }

		auto* getColorBuffer() { return m_colorBuffer; }

		const auto& getSceneBound() const { return sceneBound; }

		uint64_t getTotalSamplesComputed() const { return static_cast<uint64_t>(m_samplesComputed)*static_cast<uint64_t>(m_rayCount); }


		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxDimensions = 4u;
    protected:
        ~Renderer();


        nbl::video::IVideoDriver* m_driver;

		nbl::asset::IAssetManager* m_assetManager;
		nbl::scene::ISceneManager* m_smgr;

		nbl::core::vector<std::array<nbl::core::vector3df_SIMD,3> > m_precomputedGeodesic;

		nbl::core::smart_refctd_ptr<nbl::ext::RadeonRays::Manager> m_rrManager;

		bool m_rightHanded;

		nbl::core::smart_refctd_ptr<nbl::video::ITextureBufferObject> m_sampleSequence;
		nbl::core::smart_refctd_ptr<nbl::video::ITexture> m_scrambleTexture;

		nbl::video::E_MATERIAL_TYPE nonInstanced;
		uint32_t m_raygenProgram, m_compostProgram;
		nbl::core::smart_refctd_ptr<nbl::video::ITexture> m_depth,m_albedo,m_normals,m_lightIndex,m_accumulation,m_tonemapOutput;
		nbl::video::IFrameBuffer* m_colorBuffer,* m_gbuffer,* tmpTonemapBuffer;

		uint32_t m_maxSamples;
		uint32_t m_workGroupCount[2];
		uint32_t m_samplesPerDispatch;
		uint32_t m_samplesComputed;
		uint32_t m_rayCount;
		uint32_t m_framesDone;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_rayBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_intersectionBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_rayCountBuffer;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_rayBufferAsRR;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_intersectionBufferAsRR;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_rayCountBufferAsRR;

		nbl::core::vector<nbl::core::smart_refctd_ptr<nbl::scene::IMeshSceneNode> > nodes;
		nbl::core::aabbox3df sceneBound;
		nbl::ext::RadeonRays::Manager::MeshBufferRRShapeCache rrShapeCache;
		nbl::ext::RadeonRays::Manager::MeshNodeRRInstanceCache rrInstances;

		nbl::core::vectorSIMDf constantClearColor;
		uint32_t m_lightCount;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_lightCDFBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_lightBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_lightRadianceBuffer;
};

#endif
