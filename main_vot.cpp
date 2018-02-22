#include <stdlib.h>

#include "kcf.h"
#include "vot.hpp"

int main(int argc, char *argv[])
{
    //load region, images and prepare for output
    std::string region, images, output;

    if (argc > 1 && (argv[1] == std::string("-h") || argv[1] == std::string("--help")))
        argc = -1;

    switch (argc) {
    case 1:
        region = "region.txt";
        images = "images.txt";
        output = "output.txt";
        break;
    case 2:
        region = std::string(argv[1]) + "/region.txt";
        images = std::string(argv[1]) + "/images.txt";
        output = std::string(argv[1]) + "/output.txt";
        break;
    case 4:
        region = std::string(argv[1]);
        images = std::string(argv[2]);
        output = std::string(argv[3]);
        break;
    default:
        std::cerr << "Usage: \n"
                  << argv[0] << "\n"
                  << argv[0] << " <directory>\n"
                  << argv[0] << " <path/to/region.txt> <path/to/images.txt> <path/to/output.txt>\n";
        return 1;
    }
    VOT vot_io(region, images, output);

    KCF_Tracker tracker;
    cv::Mat image;

    //img = firts frame, initPos = initial position in the first frame
    cv::Rect init_rect = vot_io.getInitRectangle();
    vot_io.outputBoundingBox(init_rect);
    vot_io.getNextImage(image);

    tracker.init(image, init_rect);

    BBox_c bb;
    double avg_time = 0.;
    int frames = 0;
    while (vot_io.getNextImage(image) == 1){
        double time_profile_counter = cv::getCPUTickCount();
        tracker.track(image);
        time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
         std::cout << "  -> speed : " <<  time_profile_counter/((double)cvGetTickFrequency()*1000) << "ms. per frame" << std::endl;
        avg_time += time_profile_counter/((double)cvGetTickFrequency()*1000);
        frames++;

        bb = tracker.getBBox();
        vot_io.outputBoundingBox(cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h));
#ifdef VISULIZE_RESULT
       cv::rectangle(image, cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h), CV_RGB(0,255,0), 2);
       cv::imshow("output", image);
       cv::waitKey();
#endif //VISULIZE

//        std::stringstream s;
//        std::string ss;
//        int countTmp = frames;
//        s << "imgs" << "/img" << (countTmp/10000);
//        countTmp = countTmp%10000;
//        s << (countTmp/1000);
//        countTmp = countTmp%1000;
//        s << (countTmp/100);
//        countTmp = countTmp%100;
//        s << (countTmp/10);
//        countTmp = countTmp%10;
//        s << (countTmp);
//        s << ".jpg";
//        s >> ss;
//        //set image output parameters
//        std::vector<int> compression_params;
//        compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
//        compression_params.push_back(90);
//        cv::imwrite(ss.c_str(), image, compression_params);
    }

    std::cout << "Average processing speed " << avg_time/frames <<  "ms. (" << 1./(avg_time/frames)*1000 << " fps)" << std::endl;

    return EXIT_SUCCESS;
}
