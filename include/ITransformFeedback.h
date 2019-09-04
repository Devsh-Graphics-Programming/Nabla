// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_TRANSFORM_FEEDBACK_H_INCLUDED__
#define __I_TRANSFORM_FEEDBACK_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/core/parallel/IThreadBound.h"

namespace irr
{
namespace video
{

class IGPUBuffer;


class ITransformFeedback : public virtual core::IReferenceCounted, public core::IThreadBound
{
    public:
        ITransformFeedback() : active(false) {}

        virtual bool rebindRevalidate() = 0;

        virtual bool bindOutputBuffer(const size_t& index, IGPUBuffer* buffer, const size_t& offset=0, const size_t& length=0) = 0;

        virtual const IGPUBuffer* getOutputBuffer(const size_t &ix) const = 0;

        virtual size_t getOutputBufferOffset(const size_t &ix) const = 0;

        //! Will not take place until material is set which matches
		virtual void pauseTransformFeedback() = 0;

		virtual void resumeTransformFeedback() = 0;

		inline const bool& isActive() const {return active;}

    protected:
        bool active;
};


} // end namespace video
} // end namespace irr

#endif


