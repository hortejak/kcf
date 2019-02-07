#ifndef VIDEOIO_HPP
#define VIDEOIO_HPP

#include <opencv2/opencv.hpp>

class VideoIO {
public:
    virtual ~VideoIO() {}
    virtual cv::Rect getInitRectangle() const = 0;
    virtual void outputBoundingBox(const cv::Rect & bbox) = 0;
    virtual int getNextFileName(char * fName) = 0;
    virtual int getNextImage(cv::Mat & img) = 0;
};

class FileIO : public VideoIO {
public:
    FileIO(std::string video_in);

    cv::Rect getInitRectangle() const override;
    void outputBoundingBox(const cv::Rect & bbox) override;
    int getNextFileName(char * fName) override;
    int getNextImage(cv::Mat & img) override;

private:
    cv::VideoCapture capture;
    std::string filename;
};

#endif // VIDEOIO_HPP
