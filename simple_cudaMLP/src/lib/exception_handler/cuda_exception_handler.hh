/*
 *   Author of this header file: https://github.com/pwlnk
 *   Helper class to handle CUDA exceptions
 *
 */

#pragma once

#include <exception>
#include <iostream>

class NNException : std::exception 
{

public:
    NNException(const char* exception_message) 
        : exception_message{exception_message}
    {
    }

    virtual const char* what() const throw()
    {
        return exception_message;
    }

    static void throwIfDeviceErrorsOccurred(const char* exception_message) 
    {
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) 
        {
            std::cerr << error << ": " << exception_message;
            throw NNException(exception_message);
        }
    }

private:
    const char* exception_message;
};