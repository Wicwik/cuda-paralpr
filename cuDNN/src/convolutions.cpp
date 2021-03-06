#include <iostream>
#include <cudnn.h>

#include "lib/error_handler.hh"
#include "lib/image_parser.hh"

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "Invalid number of parameters\n";
    return 1;
  }

	cudnnHandle_t cudnn; // create handler struct

	cv::Mat img = ip::load_img(argv[1]);
  cudnn_check(cudnnCreate(&cudnn)); // initialize the handler

	size_t batch_size = 1; // single image
	size_t channels = 3; // RGB
	size_t height = img.rows;
	size_t width = img.cols;


  cudnnTensorDescriptor_t input_desc; // create descriptor struct
  cudnn_check(cudnnCreateTensorDescriptor(&input_desc)); // inicialize descriptor
  cudnn_check(cudnnSetTensor4dDescriptor(input_desc, // descriptor to set
  									   CUDNN_TENSOR_NHWC, // descriptor format (NCHW/NHWC)
  									   CUDNN_DATA_FLOAT, // data type
  									   batch_size, // N - batch size
  									   channels, // C - number of channels (RGB)
  									   height, // H - height
  									   width  // W - width
  									   )); 

  cudnnTensorDescriptor_t output_desc; // create descriptor struct
  cudnn_check(cudnnCreateTensorDescriptor(&output_desc)); // inicialize descriptor
  cudnn_check(cudnnSetTensor4dDescriptor(output_desc, // descriptor to set
  									   CUDNN_TENSOR_NHWC, // descriptor format (NCHW/NHWC)
  									   CUDNN_DATA_FLOAT, // data type
  									   batch_size, // N - batch size
  									   channels, // C - number of channels (RGB)
  									   height, // H - height
  									   width  // W - width
  									   )); 

  cudnnFilterDescriptor_t filter_desc; // create descriptor struct
  cudnn_check(cudnnCreateFilterDescriptor(&filter_desc)); // inicialize descriptor
  cudnn_check(cudnnSetFilter4dDescriptor(filter_desc, // descriptor to set
  									   CUDNN_DATA_FLOAT, // data type
  									   CUDNN_TENSOR_NCHW, // tensor format, we use NCHW for easier template initialization
  									   channels, // output channels
  									   channels, // input channels
  									   3, // filter height
  									   3  // filter width
  									   )); 

  cudnnConvolutionDescriptor_t convolution_desc;
  cudnn_check(cudnnCreateConvolutionDescriptor(&convolution_desc));
  cudnn_check(cudnnSetConvolution2dDescriptor(convolution_desc,
  											1, // padding height
  											1, // padding width
  											1, // vertical stride
  											1, // horizontal stride
  											1, // dilation height
  											1, // dilation width
  											CUDNN_CROSS_CORRELATION, // convolution algortithm
  											CUDNN_DATA_FLOAT // data type
  											));


  // still choosing an algo? cuDNN will do it for ya pal!
  int algo_count;
  cudnnConvolutionFwdAlgoPerf_t convolution_algorithms;
  cudnn_check(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
  												input_desc,
  												filter_desc,
  												convolution_desc,
  												output_desc,
  												0, // maximum number of available algorithms
  												&algo_count, // number of available algorithms
  												&convolution_algorithms // algorithm output
  												));

  cudnnConvolutionFwdAlgo_t convolution_algorithm = convolution_algorithms.algo; // get the algorithm

  // calculate workspace size
  size_t workspace_size = 0;
  cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
  													input_desc,
  													filter_desc,
  													convolution_desc,
  													output_desc,
  													convolution_algorithm,
  													&workspace_size	
  													));

	std::cout << "Workspace size: " << (workspace_size/1048576.0) << " MB" << std::endl;


  // memory allocation part
	void* d_workspace{nullptr};
	cudaMalloc(&d_workspace, workspace_size); // allocate workspace on device

	int image_size = batch_size * channels * height * width * sizeof(float);

  // allocate and copy input image to device
	float* d_input{nullptr};
	cudaMalloc(&d_input, image_size);
	cudaMemcpy(d_input, img.ptr<float>(0), image_size, cudaMemcpyHostToDevice);

  // allocate and set output image to zeros to device
	float *d_output{nullptr};
	cudaMalloc(&d_output, image_size);
	cudaMemset(d_output, 0, image_size);

  // filter template
	const float filer_template[3][3] =
	{
		{1,  1, 1},
		{1, -8, 1},
		{1,  1, 1}
	};

	float h_filter[channels][channels][3][3];
	for (int i = 0; i < channels; i++)
	{
		for (int j = 0; j < channels; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					h_filter[i][j][k][l] = filer_template[k][l]; // this is why we used NCHW
				}
			}
		}
	}

  // allocate and initilaze the filter on device
	float *d_filter{nullptr};
	cudaMalloc(&d_filter, sizeof(h_filter));
	cudaMemcpy(d_filter, h_filter, sizeof(h_filter), cudaMemcpyHostToDevice);

  // run convolution
	const float alpha = 1, beta = 0; // from api docs: Pointers to scaling factors (in host memory) used to blend the computation result with prior value in the output layer as follows: 
	// dstValue = alpha[0]*result + beta[0]*priorDstValue
  
  cudnn_check(cudnnConvolutionForward(cudnn,
										&alpha,
										input_desc,
										d_input,
										filter_desc,
										d_filter,
										convolution_desc,
										convolution_algorithm,
										d_workspace,
										workspace_size,
										&beta,
										output_desc,
										d_output
										));

  // allocate and copy output to host
	float *h_output = new float[image_size];
	cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

  // show image
	ip::show_img(h_output, height, width);

  // cleaning
	delete[] h_output;
	cudaFree(d_filter);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_workspace);

	cudnnDestroyTensorDescriptor(input_desc);
	cudnnDestroyTensorDescriptor(output_desc);
	cudnnDestroyFilterDescriptor(filter_desc);
	cudnnDestroyConvolutionDescriptor(convolution_desc);

  cudnnDestroy(cudnn);
}