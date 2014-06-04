#ifndef DR_HH
# define DR_HH
# include <cv.h>
# include <cvaux.h>
# include <highgui.h>

# include <list>
# include <iostream>
# include <time.h>

// FIXME: template seems cool to handle the differents types but the specifications break every chances of genericity.
// Multi-class suits our purpose.
// Handle only images for the moment and think of a good structure for handling videos, images and different contours at the same time.

// Class for Diminished reality purposes.
// Handle images and video streams (online and offline).
// Contour can be contours or colors only.
class DR
{
  public:
    // Constructor with the input and the contour.
    DR(char* mask, char* input);
    void inpaint();

  private:
    void offset_scaling_up();
    // From a binary mask, build the list of points.
    // FIXME: initialization of the mapping during the mask generation.
    void build_mask();
    double spatial_cost(cv::Point& p);
    // TODO: use a LAB or YUV image to do the appearance or weightning.
    double appearance_cost(cv::Point& p);
    double cost_bullshit(cv::Point& p, double curr_cost, bool& stop);
    void random_search(cv::Point& p, double& curr_cost);
    double weightning(cv::Point& p);
    cv::Mat FindPatch(int x, int y, cv::Mat image);
    void improve(cv::Point p, size_t cpt, double& cost);

    cv::Mat mask_;
    cv::Mat input_;
    cv::Mat gray_;
    size_t nb_iter_;
    size_t iter_;
    size_t max_scale_;
    size_t max_iter_;
    int scale_iter_;
    int patch_size_;
    std::vector<cv::Mat> pyramid_image_;
    std::vector<cv::Mat> pyramid_mapping_;
    std::vector<cv::Mat> pyramid_mask_;
    std::vector<cv::Mat> pyramid_cost_;
    std::vector<int> pyramid_size_;
    std::vector<std::list<cv::Point> > pyramid_target_pixels_;
    cv::Mat res_;
};

#endif
