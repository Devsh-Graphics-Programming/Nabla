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
        
            COpenEXRImageMetadata(std::string _name, core::smart_refctd_dynamic_array<ImageInputSemantic>&& _inputs) : name(_name), m_imageInputs(std::move(_inputs)) {}

            std::string getName() const 
            { 
                return name; 
            }

            core::SRange<const ImageInputSemantic> getCommonRequiredInputs() const override
            {
                return { m_imageInputs->begin(), m_imageInputs->end() };
            }

            const char* getLoaderName() const override 
            { 
                return "CImageLoaderOpenEXR"; 
            }

        private:

            std::string name;
            core::smart_refctd_dynamic_array<ImageInputSemantic> m_imageInputs;
           
        } PACK_STRUCT;
        #include "irr/irrunpack.h"

    }   
}

#endif // __IRR_C_OPENEXR_IMAGE_METADATA_H_INCLUDED__