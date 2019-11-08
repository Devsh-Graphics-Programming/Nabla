#ifndef _EXTRA_CRAP_INCLUDED_
#define _EXTRA_CRAP_INCLUDED_

#include "irrlicht.h"


class Renderer : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		Renderer(video::IVideoDriver* _driver);

		//setData();

		//drawLoop();
    private:
        ~Renderer();

        video::IVideoDriver* m_driver;
        video::IGPUBuffer* m_histogramBuffer;
        uint32_t m_histogramProgram,m_autoExpParamProgram;
        uint32_t m_workGroupCount[2];
        uint32_t m_totalThreadCount[2];
};

#endif
