#ifndef _EXTRA_CRAP_INCLUDED_
#define _EXTRA_CRAP_INCLUDED_

#include "irrlicht.h"

#include "../../ext/RadeonRays/RadeonRays.h"


class Renderer : public irr::core::IReferenceCounted, public irr::core::InterfaceUnmovable
{
    public:
		Renderer(irr::video::IVideoDriver* _driver, irr::asset::IAssetManager* _assetManager, irr::scene::ISceneManager* _smgr);

		void init(const irr::asset::SAssetBundle& meshes, uint32_t rayBufferSize=512u*1024u*1024u);

		void deinit();

		void render();

		auto* getColorBuffer() { return m_colorBuffer; }

		const auto& getSceneBound() const { return sceneBound; }
    protected:
        ~Renderer();

        irr::video::IVideoDriver* m_driver;
		irr::video::E_MATERIAL_TYPE nonInstanced;
		uint32_t m_raygenProgram;
		irr::asset::IAssetManager* m_assetManager;
		irr::scene::ISceneManager* m_smgr;
		irr::core::smart_refctd_ptr<irr::ext::RadeonRays::Manager> m_rrManager;

		irr::core::smart_refctd_ptr<irr::video::ITexture> m_depth,m_albedo,m_normals;
		irr::video::IFrameBuffer* m_colorBuffer,* m_gbuffer;

		uint32_t m_workGroupCount[2];
		uint32_t m_samplesPerDispatch;
		uint32_t m_rayCount;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_rayBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_intersectionBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_rayCountBuffer;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_rayBufferAsRR;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_intersectionBufferAsRR;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_rayCountBufferAsRR;

		irr::core::vector<irr::core::smart_refctd_ptr<irr::scene::IMeshSceneNode> > nodes;
		irr::core::aabbox3df sceneBound;
		irr::ext::RadeonRays::Manager::MeshBufferRRShapeCache rrShapeCache;
		irr::ext::RadeonRays::Manager::MeshNodeRRInstanceCache rrInstances;
};

#endif
