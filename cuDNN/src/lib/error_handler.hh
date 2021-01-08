/*
 *	 This macro checks if anny error happend at CUDAs side.
 */

#pragma once
#include <iostream>

#define cudnn_check(expression)																		  	\
{																										\
	cudnnStatus_t status = (expression); 																\
	if (status != CUDNN_STATUS_SUCCESS) 																\
	{ 																									\
		std::cerr << "Error on line " << __LINE__ << ": " << cudnnGetErrorString(status) << std::endl; 	\
		std::exit(EXIT_FAILURE); 																		\
	}																									\
}