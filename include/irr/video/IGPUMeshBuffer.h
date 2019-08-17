// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_GPU_MESH_BUFFER_H_INCLUDED__
#define __I_GPU_MESH_BUFFER_H_INCLUDED__

#include <algorithm>

#include "irr/asset/asset.h"

#include "ITransformFeedback.h"
#include "IGPUBuffer.h"
#include "vectorSIMD.h"

namespace irr
{
namespace video
{

	class IGPUMeshDataFormatDesc : public asset::IMeshDataFormatDesc<video::IGPUBuffer>
	{
	};


	class IGPUMeshBuffer : public asset::IMeshBuffer<video::IGPUBuffer>
	{
            video::ITransformFeedback* attachedXFormFeedback;
            uint32_t attachedXFormFeedbackStream;
        protected:
            virtual ~IGPUMeshBuffer()
            {
                if (attachedXFormFeedback)
                    attachedXFormFeedback->drop();
            }
        public:
            IGPUMeshBuffer(core::CLeakDebugger* dbgr=NULL) : asset::IMeshBuffer<video::IGPUBuffer>(NULL,dbgr), attachedXFormFeedback(NULL), attachedXFormFeedbackStream(0) {}

            inline void setIndexCountFromXFormFeedback(video::ITransformFeedback* xformFeedback, const uint32_t & stream)
            {
                attachedXFormFeedbackStream = stream;


                if (xformFeedback==attachedXFormFeedback)
                    return;

                if (!xformFeedback)
                {
                    if (attachedXFormFeedback)
                        attachedXFormFeedback->drop();

                    attachedXFormFeedback = NULL;
                    return;
                }

                xformFeedback->grab();
                if (attachedXFormFeedback)
                    attachedXFormFeedback->drop();
                attachedXFormFeedback = xformFeedback;

                indexType = asset::EIT_UNKNOWN;
                indexCount = 0;
            }

            inline video::ITransformFeedback* getXFormFeedback() const {return attachedXFormFeedback;}

            inline const uint32_t& getXFormFeedbackStream() const {return attachedXFormFeedbackStream;}

            bool isIndexCountGivenByXFormFeedback() const {return attachedXFormFeedback!=NULL;}
	};

} // end namespace video
} // end namespace irr



#endif


