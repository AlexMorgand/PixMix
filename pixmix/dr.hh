#ifndef DR_HH
# define DR_HH
# include <cv.h>
# include <cvaux.h>
# include <highgui.h>

# include <list>
# include <iostream>
# include <time.h>


// Class for Diminished reality purposes.
// Handle images and video streams (online and offline).
// Contour can be contours or colors only.

class DR
{
  public:
    // Policy for the patch copying.
    enum COPY_P
    {
        SIMPLE,
        INTENSITY
    };

    // Constructor with the input and the contour.
    DR(char* mask, char* input, COPY_P cp = SIMPLE);
    void inpaint();

  private:
    // Copy the previous layer of the pyramid.
    void offset_scaling_up();

    // From a binary mask, build the list of points.
    void build_mask();

    // Basic cost comparing patches in the LAB colorspace.
    double cost_bullshit(cv::Point& p, double curr_cost, bool& stop);

    void random_search(cv::Point& p, double& curr_cost);

    void improve(cv::Point p, size_t cpt, double& cost);

    // Copy the patch or raise intensity.
    void patch_copy(cv::Point dst, cv::Point src);

    COPY_P cp_;
    cv::Mat mask_;
    cv::Mat input_;
    size_t nb_iter_;
    size_t iter_;
    size_t max_scale_;
    size_t max_iter_;
    int scale_iter_;
    int patch_size_;

    // Pyramid attributes.

    std::vector<cv::Mat> pyramid_image_;
    std::vector<cv::Mat> pyramid_mapping_;
    std::vector<cv::Mat> pyramid_mask_;
    std::vector<cv::Mat> pyramid_cost_;

    // Size of patches for every layers of the pyramid.
    std::vector<int> pyramid_size_;

    std::vector<std::list<cv::Point> > pyramid_target_pixels_;
    cv::Mat res_;
};

#endif
