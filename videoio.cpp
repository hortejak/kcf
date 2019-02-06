#include "videoio.hpp"


FileIO::FileIO(std::string video_in)
    : capture(video_in)
{
    if (!capture.isOpened())
        throw std::runtime_error("Cannot open video stream '" + video_in + "'");
}

cv::Rect FileIO::getInitRectangle() const
{
    return cv::Rect();
}

void FileIO::outputBoundingBox(const cv::Rect &bbox)
{
    (void)bbox;
}

int FileIO::getNextFileName(char *fName)
{
    (void)fName;
    return 0;
}

int FileIO::getNextImage(cv::Mat &img)
{
    capture >> img;
    return img.empty() ? 0 : 1;
}
