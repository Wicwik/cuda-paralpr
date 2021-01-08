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
        : _exception_message{exception_message}
    {
    }

    virtual const char* what() const throw()
    {
        return _exception_message;
    }

    static void throwIfDeviceErrorsOccurred(const char* exception_message) 
    {
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) 
        {
            std::cerr << error << ": " << exception_message;
            throw NNException(exception_message); // throw and exception if an error happens
        }
    }

private:
    const char* _exception_message;
};