#include <stdlib.h>
#include <getopt.h>
#include <libgen.h>
#include <unistd.h>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <memory>
#include <err.h>

#include "kcf.h"
#include "vot.hpp"
#include "videoio.hpp"

// Needed for OpenCV <= 3.2 as replacement for Rect::empty()
bool empty(cv::Rect r)
{
    return r.width <= 0 || r.height <= 0;
}

double calcAccuracy(std::string line, cv::Rect bb_rect, cv::Rect &groundtruth_rect)
{
    std::vector<float> numbers;
    std::istringstream s(line);
    float x;
    char ch;

    while (s >> x) {
        numbers.push_back(x);
        s >> ch;
    }
    double x1 = std::min(numbers[0], std::min(numbers[2], std::min(numbers[4], numbers[6])));
    double x2 = std::max(numbers[0], std::max(numbers[2], std::max(numbers[4], numbers[6])));
    double y1 = std::min(numbers[1], std::min(numbers[3], std::min(numbers[5], numbers[7])));
    double y2 = std::max(numbers[1], std::max(numbers[3], std::max(numbers[5], numbers[7])));

    groundtruth_rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);

    double rects_intersection = (groundtruth_rect & bb_rect).area();
    double rects_union = (groundtruth_rect | bb_rect).area();
    double accuracy = rects_intersection / rects_union;

    return accuracy;
}

int main(int argc, char *argv[])
{
    //load region, images and prepare for output
    std::string region, images, output, video_out;
    int visualize_delay = -1, fit_size_x = -1, fit_size_y = -1;
    KCF_Tracker tracker;
    cv::VideoWriter videoWriter;
    cv::Rect init_rect;
    bool set_box_interactively = false;

    while (1) {
        int option_index = 0;
        static struct option long_options[] = {
            {"debug",     no_argument,       0,  'd' },
            {"visual_debug", optional_argument,    0, 'p'},
            {"help",      no_argument,       0,  'h' },
            {"output",    required_argument, 0,  'o' },
            {"video_out", optional_argument, 0,  'O' },
            {"visualize", optional_argument, 0,  'v' },
            {"fit",       optional_argument, 0,  'f' },
            {"box",       optional_argument, 0,  'b' },
            {0,           0,                 0,  0 }
        };

        int c = getopt_long(argc, argv, "b::dp::hv::f::o:O::", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
        case 'b':
            if (optarg) {
                if (sscanf(optarg, "%d,%d,%d,%d", &init_rect.x, &init_rect.y, &init_rect.width, &init_rect.height) != 4)
                    errx(1, "Invalid box specification: %s", optarg);
            } else {
                set_box_interactively = true;
                errx(1, "Interactive box specification not implemented");
            }
            break;
        case 'd':
            tracker.m_debug = true;
            break;
        case 'p':
            if (!optarg || *optarg == 'p')
                tracker.m_visual_debug = KCF_Tracker::vd::PATCH;
            else if (optarg && *optarg == 'r')
                tracker.m_visual_debug = KCF_Tracker::vd::RESPONSE;
            else {
                fprintf(stderr, "Unknown visual debug mode: %c", *optarg);
                return 1;
            }
            break;
        case 'h':
            std::cerr << "Usage: \n"
                      << argv[0] << " [options]\n"
                      << argv[0] << " [options] <directory>\n"
                      << argv[0] << " [options] <video_file>\n"
                      << argv[0] << " [options] <path/to/region.txt or groundtruth.txt> <path/to/images.txt> [path/to/output.txt]\n"
                      << "Options:\n"
                      << " --visualize    | -v[delay_ms]\n"
                      << " --output       | -o <output.txt>\n"
                      << " --fit          | -f[W[xH]]\n"
                      << " --debug        | -d\n"
                      << " --visual_debug | -p [p|r]\n"
                      << " --box          | -b [X,Y,W,H]\n";
            exit(0);
            break;
        case 'o':
            output = optarg;
            break;
        case 'O':
            video_out = optarg ? optarg : "./output.avi";
            break;
        case 'v':
            visualize_delay = optarg ? atol(optarg) : 1;
            break;
        case 'f':
            if (!optarg) {
                fit_size_x = fit_size_y = 0;
            } else {
                char tail;
                if (sscanf(optarg, "%d%c", &fit_size_x, &tail) == 1) {
                    fit_size_y = fit_size_x;
                } else if (sscanf(optarg, "%dx%d%c", &fit_size_x, &fit_size_y, &tail) != 2) {
                    fprintf(stderr, "Cannot parse -f argument: %s\n", optarg);
                    return 1;
                }
            }
            break;
        }
    }

    std::unique_ptr<VideoIO> io;

    switch (argc - optind) {
    case 1:
        struct stat st;
        if (stat(argv[optind], &st) != 0) {
            perror(argv[optind]);
            exit(1);
        }
        if (S_ISDIR(st.st_mode)) {
            if (chdir(argv[optind]) == -1) {
                perror(argv[optind]);
                exit(1);
            }
        } else if (S_ISREG(st.st_mode)) {
            io.reset(new FileIO(argv[optind]));
            break;
        }
        // Fall through
    case 0:
        region = access("groundtruth.txt", F_OK) == 0 ? "groundtruth.txt" : "region.txt";
        images = "images.txt";
        if (output.empty())
            output = "output.txt";
        break;
    case 2:
        // Fall through
    case 3:
        region = std::string(argv[optind + 0]);
        images = std::string(argv[optind + 1]);
        if (output.empty()) {
            if ((argc - optind) == 3)
                output = std::string(argv[optind + 2]);
            else
                output = std::string(dirname(argv[optind + 0])) + "/output.txt";
        }
        break;
    default:
        std::cerr << "Too many arguments\n";
        return 1;
    }

    if (!io)
        io.reset(new VOT(region, images, output));

    // if groundtruth.txt is used use intersection over union (IOU) to calculate tracker accuracy
    std::ifstream groundtruth_stream;
    if (region.compare("groundtruth.txt") == 0) {
        groundtruth_stream.open(region.c_str());
        std::string line;
        std::getline(groundtruth_stream, line);
    }

    cv::Mat image;

    //img = firts frame, initPos = initial position in the first frame
    if (empty(init_rect))
        init_rect = io->getInitRectangle(); // Try to get BBox from VOT files
    io->outputBoundingBox(init_rect);
    io->getNextImage(image);

    if (!video_out.empty()) {
        int codec = CV_FOURCC('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
        double fps = 25.0;                          // framerate of the created video stream
        videoWriter.open(video_out, codec, fps, image.size(), true);
    }

    tracker.init(image, init_rect, fit_size_x, fit_size_y);


    BBox_c bb;
    cv::Rect bb_rect;
    double avg_time = 0., sum_accuracy = 0.;
    int frames = 0;

    std::cout << std::fixed << std::setprecision(2);

    while (io->getNextImage(image) == 1){
        double time_profile_counter = cv::getCPUTickCount();
        tracker.track(image);
        time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
         std::cout << "  -> speed : " <<  time_profile_counter/((double)cvGetTickFrequency()*1000) << "ms per frame, "
                      "response : " << tracker.getFilterResponse();
        avg_time += time_profile_counter/((double)cvGetTickFrequency()*1000);
        frames++;

        bb = tracker.getBBox();
        bb_rect = cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h);
        io->outputBoundingBox(bb_rect);

        if (groundtruth_stream.is_open()) {
            std::string line;
            std::getline(groundtruth_stream, line);

            cv::Rect groundtruthRect;
            double accuracy = calcAccuracy(line, bb_rect, groundtruthRect);
            if (visualize_delay >= 0)
                cv::rectangle(image, groundtruthRect, CV_RGB(255, 0,0), 1);
            std::cout << ", accuracy: " << accuracy;
            sum_accuracy += accuracy;
        }

        std::cout << std::endl;

        if (visualize_delay >= 0 || !video_out.empty()) {
            cv::Point pt(bb.cx, bb.cy);
            cv::Size size(bb.w, bb.h);
            cv::RotatedRect rotatedRectangle(pt, size, bb.a);

            cv::Point2f vertices[4];
            rotatedRectangle.points(vertices);

            for (int i = 0; i < 4; i++)
                cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            if (visualize_delay >= 0) {
                cv::imshow("KCF output", image);
                int ret = cv::waitKey(visualize_delay);
                if ((visualize_delay > 0 && ret != -1 && ret < 128) ||
                        (visualize_delay == 0 && (ret == 27 /*esc*/ || ret == 'q')))
                    break;
            }
            if (!video_out.empty())
                videoWriter << image;
        }

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

    std::cout << "Average processing speed: " << avg_time / frames << "ms (" << 1. / (avg_time / frames) * 1000 << " fps)";
    if (groundtruth_stream.is_open()) {
        std::cout << "; Average accuracy: " << sum_accuracy/frames << std::endl;
        groundtruth_stream.close();
    }
    if (!video_out.empty())
       videoWriter.release();
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
