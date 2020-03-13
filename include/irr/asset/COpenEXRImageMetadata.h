#ifndef __IRR_C_OPENEXR_IMAGE_METADATA_H_INCLUDED__
#define __IRR_C_OPENEXR_IMAGE_METADATA_H_INCLUDED__

#include "irr/asset/IImageMetadata.h"

namespace irr 
{
    namespace asset
    {
        #include "irr/irrpack.h"
        class COpenEXRImageMetadata final : public IImageMetadata
        {
        public:
        
            COpenEXRImageMetadata(core::smart_refctd_dynamic_array<ImageInputSemantic>&& _inputs) : m_imageInputs(std::move(_inputs)) {}

            core::SRange<const ImageInputSemantic> getCommonRequiredInputs() const override
            {
                return { m_imageInputs->begin(), m_imageInputs->end() };
            }

            const char* getLoaderName() const override 
            { 
                return "CImageLoaderOpenEXR"; 
            }

        private:

            core::smart_refctd_dynamic_array<ImageInputSemantic> m_imageInputs;
           
        } PACK_STRUCT;
        #include "irr/irrunpack.h"

    }   
}

#endif // __IRR_C_OPENEXR_IMAGE_METADATA_H_INCLUDED__