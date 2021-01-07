#include <opencv2/opencv.hpp>

cv::Mat load_image(const char* image_path) 
{
  cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  return image;
}

int main()
{
    return 0;
}