#pragma once

#include <opencv2/opencv.hpp>

namespace ip
{ 
    cv::Mat load_img(std::string image_path)
    {
        cv::Mat img = imread(image_path, cv::IMREAD_COLOR);

        img.convertTo(img, CV_32FC3);
      	cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);

        return img;
    }

    void show_img(float* buffer, int height, int width, std::string save_path = "./saved_img.png")
    {
        cv::Mat output_image(height, width, CV_32FC3, buffer);
        cv::threshold(output_image, output_image, 0, 0, cv::THRESH_TOZERO);
        cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
        output_image.convertTo(output_image, CV_8UC3);

    	cv::imshow("Display window", output_image);
        int k = cv::waitKey(0); // Wait for a keystroke in the window
        
        if(k == 's')
        {
            cv::imwrite(save_path, output_image);
        }
    }
}