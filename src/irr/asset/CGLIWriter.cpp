/*
MIT License
Copyright (c) 2019 AnastaZIuk
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "CGLIWriter.h"

#ifdef _IRR_COMPILE_WITH_GLI_WRITER_

#ifdef _IRR_COMPILE_WITH_GLI_
#include "gli/gli.hpp"
#else
#error "It requires GLI library"
#endif

namespace irr
{
	namespace asset
	{
		bool CGLIWriter::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
		{
			if (!_override)
				getDefaultOverride(_override);

			SAssetWriteContext ctx{ _params, _file };

			const asset::ICPUImage* image =
			#ifndef _IRR_DEBUG
				static_cast<const asset::ICPUImage*>(_params.rootAsset);
			#else
				dynamic_cast<const asset::ICPUImage*>(_params.rootAsset);
			#endif
			assert(image);

			io::IWriteFile* file = _override->getOutputFile(_file, ctx, { image, 0u });

			if (!file)
				return false;

			os::Printer::log("Writing GLI image", file->getFileName().c_str());

			return true;
		}

		bool CGLIWriter::writeGLIFile(io::IWriteFile* file, const asset::ICPUImage* image)
		{
			return true; // to implement after dealing with loader
		}
	}
}

#endif // _IRR_COMPILE_WITH_GLI_WRITER_