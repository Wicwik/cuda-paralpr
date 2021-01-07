#include <opencv2/opencv.hpp>

using namespace cv;

const std::string IPATH = "../img/nvidia_logo.png";

Mat load_img(std::string image_path)
{
    Mat img = imread(image_path, IMREAD_COLOR);

    return img;
}

void show_img(Mat img)
{
	imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite(IPATH, img);
    }
}

int main()
{
	Mat img = load_img(IPATH);

	if(img.empty())
    {
        std::cout << "Could not read the image: " << IPATH << std::endl;
        return 1;
    }

    show_img(img);

	return 0;
}