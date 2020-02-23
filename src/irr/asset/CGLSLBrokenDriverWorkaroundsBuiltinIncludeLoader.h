#ifndef __IRR_C_GLSL_BROKEN_DRIVER_WORKAROUNDS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_BROKEN_DRIVER_WORKAROUNDS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr 
{
	namespace asset
	{

		class CGLSLBrokenDriverWorkaroundsBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
		{
		public:
			const char* getVirtualDirectoryName() const override { return "glsl/broken_driver_workarounds/"; }

		private:
			static std::string getirr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(const std::string&)
			{
				return
					R"(
						#ifndef _IRR_BROKEN_DRIVER_WORKAROUNDS_AMD_INCLUDED_
						#define _IRR_BROKEN_DRIVER_WORKAROUNDS_AMD_INCLUDED_

                        // howabout you fix your fucking drivers AMD? Reported 4 years ago, and still same bug! And I thought it's impossible to be worse than Intel!

						dmat2x2 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_dmat2x2(dmat2x2 dummy) {return dummy;}
						dmat2x3 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_dmat2x3(dmat2x3 dummy) {return dummy;}
						dmat2x4 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_dmat2x4(dmat2x4 dummy) {return dummy;}
						dmat3x2 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_dmat3x2(dmat3x2 dummy) {return dummy;}
						dmat3x3 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_dmat3x3(dmat3x3 dummy) {return dummy;}
						dmat3x4 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_dmat3x4(dmat3x4 dummy) {return dummy;}
						dmat4x2 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_dmat4x2(dmat4x2 dummy) {return dummy;}
						dmat4x3 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_dmat4x3(dmat4x3 dummy) {return dummy;}
						dmat4x4 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_dmat4x4(dmat4x4 dummy) {return dummy;}

						mat2x2 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat2x2(mat2x2 dummy) {return dummy;}
						mat2x3 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat2x3(mat2x3 dummy) {return dummy;}
						mat2x4 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat2x4(mat2x4 dummy) {return dummy;}
						mat3x2 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat3x2(mat3x2 dummy) {return dummy;}
						mat3x3 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat3x3(mat3x3 dummy) {return dummy;}
						mat3x4 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat3x4(mat3x4 dummy) {return dummy;}
						mat4x2 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x2(mat4x2 dummy) {return dummy;}
						mat4x3 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x3(mat4x3 dummy) {return dummy;}
						mat4x4 irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x4(mat4x4 dummy) {return dummy;}

						#endif
					)";
			}

		protected:
			irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
			{
				return {
					{ std::regex{"amd\\.glsl"}, &getirr_builtin_glsl_workaround_AMD_broken_row_major_qualifier },
				};
			}
		};

	}
}

#endif // __IRR_C_GLSL_BROKEN_DRIVER_WORKAROUNDS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__