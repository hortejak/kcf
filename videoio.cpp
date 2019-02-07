#include "videoio.hpp"
#include <iostream>


FileIO::FileIO(std::string video_in)
    : capture(video_in), filename(video_in)
{
    if (!capture.isOpened())
        throw std::runtime_error("Cannot open video stream '" + video_in + "'");
}

cv::Rect FileIO::getInitRectangle() const
{
    size_t lastdot = filename.find_last_of(".");
    std::string txt = (lastdot == std::string::npos) ? filename : filename.substr(0, lastdot);
    txt += ".txt";

    FILE *f = fopen(txt.c_str(), "r");
    if (!f) {
        std::cout << txt << " not found - using empty init rectangle" << std::endl;
        return cv::Rect();
    }
    cv::Rect r;
    if (fscanf(f, "%d,%d,%d,%d", &r.x, &r.y, &r.width, &r.height) != 4) {
        std::cerr << "Error reading init rectangle from " << txt << std::endl;
        return cv::Rect();
    }
    return r;
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
