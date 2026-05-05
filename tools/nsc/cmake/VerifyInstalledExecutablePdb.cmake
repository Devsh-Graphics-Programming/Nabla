if(NOT DEFINED NBL_CONFIG)
	message(FATAL_ERROR "NBL_CONFIG is not set.")
endif()

if(NOT DEFINED NBL_PDB_FILEPATH)
	message(FATAL_ERROR "NBL_PDB_FILEPATH is not set.")
endif()

if(NOT NBL_CONFIG STREQUAL "Debug")
	return()
endif()

if(NOT EXISTS "${NBL_PDB_FILEPATH}")
	message(FATAL_ERROR "Expected installed NSC PDB at \"${NBL_PDB_FILEPATH}\".")
endif()
