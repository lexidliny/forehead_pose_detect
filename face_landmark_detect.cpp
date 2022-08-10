#include <headDetect.h>
#include <IrisLandmark.hpp>



int main(int argc,char** argv)
{

    my::IrisLandmark irisLandmarker("/home/yxw/Project/blood_collection/face_landmark/models");
    std::cout << "test" << std::endl;
    rs2::pipeline pipe;
    rs2::config cfg;
    std::cout << "test" << std::endl;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);
    std::cout << "test" << std::endl;
    while(1)
    {

        rs2::frameset frames = pipe.wait_for_frames();
        rs2::align align_to_color(RS2_STREAM_COLOR);
        rs2::frameset alignedframe = align_to_color.process(frames);

        auto depth = alignedframe.get_depth_frame();
        auto color = alignedframe.get_color_frame();

        const int w_c = color.as<rs2::video_frame>().get_width();
        const int h_c = color.as<rs2::video_frame>().get_height();

        cv::Mat image_color(cv::Size(w_c, h_c), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat image_depth(cv::Size(w_c, h_c), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);

        MM::head::headDetect dec1;

        dec1.setImage(image_color, image_depth);
        dec1.getheadPoint(irisLandmarker);
        dec1.showDetectedImage();
        if (cv::waitKey(1) == 112) // low case 'p' key
        {
            dec1.setPointCloud(color, depth);
            dec1.getheadPose(depth);
            dec1.showDetectedPointCloud();
        
            break;
        }
    }
    return 0;
}