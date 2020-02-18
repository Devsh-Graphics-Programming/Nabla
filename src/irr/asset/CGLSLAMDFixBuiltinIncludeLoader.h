#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr 
{
	namespace asset
	{

		class CGLSLAMDFixBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
		{
		public:
			const char* getVirtualDirectoryName() const override { return "glsl/amd_fix/"; }

		private:
			static std::string getAmdFix(const std::string&)
			{
				return
					R"(
						#ifndef _IRR_AMD_FIX_INCLUDED_
						#define _IRR_AMD_FIX_INCLUDED_

						dmat2 AMDFix_dmat2(dmat2 dummy) {return dummy;}
						dmat3 AMDFix_dmat3(dmat3 dummy) {return dummy;}
						dmat4 AMDFix_dmat4(dmat4 dummy) {return dummy;}

						dmat2x2 AMDFix_dmat2x2(dmat2x2 dummy) {return dummy;}
						dmat2x3 AMDFix_dmat2x3(dmat2x3 dummy) {return dummy;}
						dmat2x4 AMDFix_dmat2x4(dmat2x4 dummy) {return dummy;}
						dmat3x2 AMDFix_dmat3x2(dmat3x2 dummy) {return dummy;}
						dmat3x3 AMDFix_dmat3x3(dmat3x3 dummy) {return dummy;}
						dmat3x4 AMDFix_dmat3x4(dmat3x4 dummy) {return dummy;}
						dmat4x2 AMDFix_dmat4x2(dmat4x2 dummy) {return dummy;}
						dmat4x3 AMDFix_dmat4x3(dmat4x3 dummy) {return dummy;}
						dmat4x4 AMDFix_dmat4x4(dmat4x4 dummy) {return dummy;}

						mat2 AMDFix_mat2(mat2 dummy) {return dummy;}
						mat3 AMDFix_mat3(mat3 dummy) {return dummy;}
						mat4 AMDFix_mat4(mat4 dummy) {return dummy;}

						mat2x2 AMDFix_mat2x2(mat2x2 dummy) {return dummy;}
						mat2x3 AMDFix_mat2x3(mat2x3 dummy) {return dummy;}
						mat2x4 AMDFix_mat2x4(mat2x4 dummy) {return dummy;}
						mat3x2 AMDFix_mat3x2(mat3x2 dummy) {return dummy;}
						mat3x3 AMDFix_mat3x3(mat3x3 dummy) {return dummy;}
						mat3x4 AMDFix_mat3x4(mat3x4 dummy) {return dummy;}
						mat4x2 AMDFix_mat4x2(mat4x2 dummy) {return dummy;}
						mat4x3 AMDFix_mat4x3(mat4x3 dummy) {return dummy;}
						mat4x4 AMDFix_mat4x4(mat4x4 dummy) {return dummy;}

						#endif
					)";
			}

		protected:
			irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
			{
				return {
					{ std::regex{"amd_fix\\.glsl"}, &getAmdFix },
				};
			}
		};

	}
}