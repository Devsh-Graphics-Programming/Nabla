#ifndef _EXTRA_CRAP_INCLUDED_
#define _EXTRA_CRAP_INCLUDED_

#include "irrlicht.h"

#inlude "../../ext/RadeonRays/RadeonRays.h"


class Renderer : public irr::core::IReferenceCounted, public irr::core::InterfaceUnmovable
{
    public:
		Renderer(irr::video::IVideoDriver* _driver, uint32_t samplesPerPixel);

		//void setData();

		//void drawLoop();
    private:
        ~Renderer();

		using irr;

        video::IVideoDriver* m_driver;
		core::smart_refctd_ptr<ext::RadeonRays::Manager> m_rrManager;

        core::smart_refctd_ptr<video::IGPUBuffer> m_rayBuffer;
		void* m_rayBufferAsCL;
        uint32_t m_raygenProgram;
};

#endif
