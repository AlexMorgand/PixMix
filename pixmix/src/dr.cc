#include "dr.hh"

cv::Vec3b get_cost(unsigned x, unsigned y, cv::Mat& img)
{        
	cv::Vec3b res;
	if (x >= img.rows || x < 0 || y >= img.cols || y < 0)
		res = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
	else
		res = img.at<cv::Vec3b>(x, y);
	return res;
}

cv::Mat Scale(int numberOfScales, cv::Mat image)
{
    cv::Mat temp = image.clone();
    for (int i = 0; i < numberOfScales; i++)
        cv::resize(image, temp, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    return temp;
}

void DR::offset_scaling_up()
{
    size_t r = pyramid_image_[scale_iter_ + 1].rows - 1;
    size_t c = pyramid_image_[scale_iter_ + 1].cols - 1;

    cv::Mat tmp(pyramid_mask_[scale_iter_].size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat tmp2 = tmp.clone();
    std::list<cv::Point>::iterator it;
    for (it = pyramid_target_pixels_[scale_iter_].begin(); it != pyramid_target_pixels_[scale_iter_].end(); ++it)
    {
        cv::Point ab(it->x / 2 , it->y / 2);
        if (pyramid_mask_[scale_iter_ + 1].at<uchar>(ab.x, ab.y))
        {
            cv::Point old = pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y);

            // Multiply offset by two due to upsampling.
            cv::Point m = pyramid_mapping_[scale_iter_ + 1].at<cv::Point>(ab.x, ab.y);
            m.x *= 2.0;
            m.y *= 2.0;

            // Still in the mask.
            if (pyramid_mask_[scale_iter_].at<uchar>(it->x + m.x, it->y + m.y))
                continue;
            pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y) = m;

            // FIXME: not sure if better.
            // Random is better than the previous guess in the small scale.
            //if (pyramid_cost_[scale_iter_].at<float>(it->x, it->y) < cost)
            //    pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y) = old;
            //else
            {
                patch_copy(*it, cv::Point(m.x + it->x, m.y + it->y));
                pyramid_cost_[scale_iter_].at<float>(it->x, it->y) = pyramid_cost_[scale_iter_ + 1].at<float>(ab.x, ab.y);
                tmp.at<cv::Vec3b>(it->x, it->y) = pyramid_image_[scale_iter_].at<cv::Vec3b>(m.x + it->x, m.y + it->y);
            }
        }

    }
    cv::resize(tmp, tmp2, pyramid_image_[0].size(), 0, 0, cv::INTER_NEAREST);
    cv::imshow("scaled", tmp2);
}

DR::DR(char* mask, char* input, std::string& prefix, COPY_P cp)
  : cp_(cp),
    mask_(cv::imread(prefix + mask, CV_LOAD_IMAGE_GRAYSCALE)),
    input_(cv::imread(prefix + input)),
	prefix_(prefix),
    nb_iter_(5),
    iter_(5),
    max_scale_(4),
    scale_iter_(4),
    // PatchMatch : 9, pixmix : 5.
    patch_size_(9),
    pyramid_image_(5),
    pyramid_mapping_(5),
    pyramid_mask_(5),
    pyramid_cost_(5),
    pyramid_size_(5),
    pyramid_target_pixels_(5),
    res_(input_.size(), CV_8UC3)
{
  cv::Mat curr = input_;
  cv::Mat curr_m = mask_;
  pyramid_image_[0] = input_;
  pyramid_cost_[0] = cv::Mat(curr.size(), CV_32FC1, cv::Scalar(0, 0, 0));
  pyramid_mapping_[0] = cv::Mat(curr.size(), CV_32SC2);
  pyramid_mask_[0] = mask_;

  for (size_t i = 1; i <= max_scale_; ++i)
  {
      curr = Scale(1, curr);
      pyramid_image_[i] = curr;
	  
      curr_m = Scale(1, curr_m);
      pyramid_mask_[i] = curr_m;

      pyramid_cost_[i] = cv::Mat(curr.size(), CV_32FC1, cv::Scalar(0, 0, 0));
      pyramid_mapping_[i] = cv::Mat(curr.size(), CV_32SC2);
  }

  srand(time(0));
}

double DR::cost_bullshit(cv::Point& p, double curr_cost, bool& stop)
{
  double cost = 0;
  cv::Point tmp = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y);

  cv::Mat& tmp_input = pyramid_image_[scale_iter_];

  double d1, d2, d3;

  //FIXME: tmp is out ?!
  int mini = - (patch_size_ / 2);
  int maxi = patch_size_ / 2;
//  int mini = - pyramid_size_[scale_iter_] / 2;
//  int maxi = pyramid_size_[scale_iter_] / 2;

  for (int i = mini; i <= maxi && !stop; ++i)
  {
    for (int j = mini; j <= maxi && !stop; ++j)
    {
		cv::Vec3b pix_orig = get_cost(p.x + i, p.y + j, tmp_input);
		cv::Vec3b pix_offset = get_cost(p.x + i + tmp.x, p.y + j + tmp.y, tmp_input);
		d1 = pix_orig[0] - pix_offset[0];
		d2 = pix_orig[1] - pix_offset[1];
		d3 = pix_orig[2] - pix_offset[2];
		d1 *= d1;
        d2 *= d2;
        d3 *= d3;
        cost += d1 + d2 + d3;

		if (curr_cost < cost)
			stop = true;

    }
  }
  return cost;

}

void DR::build_mask()
{
    size_t rows = pyramid_mask_[scale_iter_].rows;
    size_t cols = pyramid_mask_[scale_iter_].cols;

	// Do not duplicate the random.
	cv::Mat used_pixels = pyramid_mask_[scale_iter_].clone();
		
    int x, y;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (pyramid_mask_[scale_iter_].at<uchar>(i, j))
            {
                pyramid_target_pixels_[scale_iter_].push_back(cv::Point(i, j));
				// Find a pixel outside of the mask.
                do
                {
                    x = rand() % (rows);
                    y = rand() % (cols);
                } while (used_pixels.at<uchar>(x, y));
				
				/*std::cout << "current point mask " << i << " " << j << std::endl;
				std::cout << "current point " << x << " " << y << std::endl;
				std::cout << "limits " << (x - patch_size_ / 2) << " " << (y - patch_size_ / 2) << " " << (x + patch_size_ / 2) << " " << (y + patch_size_ / 2) << std::endl;
				std::cout << "size " << rows << " " << cols << std::endl;*/

				used_pixels.at<uchar>(x, y) = 255;
                // Offset computing.
                cv::Point m = cv::Point(x - i, y - j);
                pyramid_mapping_[scale_iter_].at<cv::Point>(i, j) = m;

                patch_copy(cv::Point(i, j), cv::Point(m.x + i, m.y + j));
            }
            else
                pyramid_mapping_[scale_iter_].at<cv::Point>(i, j) = cv::Point(0, 0);
        }
    }
	
				
    double alpha = 0.5;
    for (auto it = pyramid_target_pixels_[scale_iter_].begin(); it != pyramid_target_pixels_[scale_iter_].end(); ++it)
    {
        // For the initialization, the current cost is not computed.
        // Put INT_MAX instead.
        double max = INT_MAX;
        bool stop = false;
		pyramid_cost_[scale_iter_].at<float>(it->x, it->y) = cost_bullshit(*it, max, stop);
    }
	cv::imshow("input", pyramid_cost_[scale_iter_]);
	cv::waitKey(0);

}

void DR::patch_copy(cv::Point dst, cv::Point src)
{
    if (cp_ == SIMPLE)
        pyramid_image_[scale_iter_].at<cv::Vec3b>(dst.x, dst.y) = pyramid_image_[scale_iter_].at<cv::Vec3b>(src.x, src.y);
    else if (cp_ == INTENSITY)
    {
        int mini = - (patch_size_ / 2);
        int maxi = patch_size_ / 2;
        //  int mini = - pyramid_size_[scale_iter_] / 2;
        //  int maxi = pyramid_size_[scale_iter_] / 2;

        int r = 0, g = 0, b = 0;
        for (int i = mini; i <= maxi; ++i)
        {
            for (int j = mini; j <= maxi; ++j)
            {
                b += pyramid_image_[scale_iter_].at<cv::Vec3b>(src.x, src.y)[0];
                g += pyramid_image_[scale_iter_].at<cv::Vec3b>(src.x, src.y)[1];
                r += pyramid_image_[scale_iter_].at<cv::Vec3b>(src.x, src.y)[2];
            }
        }
        r /= patch_size_ * patch_size_;
        g /= patch_size_ * patch_size_;
        b /= patch_size_ * patch_size_;
        r = (r + g + b ) / 3;
        pyramid_image_[scale_iter_].at<cv::Vec3b>(dst.x, dst.y)[0] = r;
        pyramid_image_[scale_iter_].at<cv::Vec3b>(dst.x, dst.y)[1] = r;
        pyramid_image_[scale_iter_].at<cv::Vec3b>(dst.x, dst.y)[2] = r;
    }
}

void DR::random_search(cv::Point& p, double& curr_cost)
{
    //float w = (max_iter_ - scale_iter_ + 1) * 50;
    float w = cv::min(pyramid_mapping_[scale_iter_].rows, pyramid_mapping_[scale_iter_]. cols) / 4;
    float alpha = 0.5;

    float alphai;
    bool stop = false;

    // Random value between [-1;1]
    float rx = rand() % 2000 / 1000.f - 1.f;
    float ry = rand() % 2000 / 1000.f - 1.f;
    int i = 0;
    float hypo;
    cv::Mat tmp_map = pyramid_mapping_[scale_iter_];

    cv::Point old = tmp_map.at<cv::Point>(p.x, p.y);
    do
    {
        alphai = pow(alpha, i);
        cv::Point pt(w * alphai * rx, w * alphai * ry);
        hypo = sqrt(pow(pt.x, 2.0) + pow(pt.y, 2.0));

        cv::Point randp(tmp_map.at<cv::Point>(p.x, p.y).x + p.x + pt.x,
                        tmp_map.at<cv::Point>(p.x, p.y).y + p.y + pt.y);

        if ((randp.x > (patch_size_ / 2) && randp.x < (tmp_map.rows - patch_size_ /*pyramid_size_[scale_iter_]*/ / 2)) &&
             (randp.y > (patch_size_ / 2) && randp.y < (tmp_map.cols - patch_size_ /*pyramid_size_[scale_iter_]*/ / 2)))
        {

            if (pyramid_mask_[scale_iter_].at<uchar>(randp.x, randp.y) == 0)
            {
                tmp_map.at<cv::Point>(p.x, p.y).x = randp.x - p.x;
                tmp_map.at<cv::Point>(p.x, p.y).y = randp.y - p.y;

                stop = false;
                double cost = cost_bullshit(p, curr_cost, stop);

                if (!stop && cost < curr_cost)
                {
                    curr_cost = cost;
                    old = tmp_map.at<cv::Point>(p.x, p.y);
                }
                else
                    tmp_map.at<cv::Point>(p.x, p.y) = old;
            }
        }

        i++;
    }
    // If below than one pixel break the loop.
    while (hypo >= 1);
}

// FIXME: Beware of the mask in the border of the images.
void DR::improve(cv::Point p, size_t cpt, double& cost)
{
    bool stop;
    bool out = false;
    // Compute cost.
    double curr_cost = pyramid_cost_[scale_iter_].at<float>(p.x, p.y);

    cv::Point old = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y);
    cv::Point newx;
    // If odd, search up.
    if (cpt % 2 != 0)
    {
        if (p.x > 0)
        {
            // Up is better ?
            newx = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x - 1, p.y);
            // Take the pixel above if you're at the border.
            if (pyramid_mask_[scale_iter_].at<uchar>(p.x - 1, p.y) == 0)
            {
                newx.x = -1;
                newx.y = 0;
            }
            else if (pyramid_mask_[scale_iter_].at<uchar>(newx.x + p.x, newx.y + p.y))
                newx.x -= 1;
        }
        else
            out = true;
    }
    // If even, search down.
    else
    {
        if (p.x < pyramid_mapping_[scale_iter_].rows - 1)
        {
            // Down is better ?
            newx = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x + 1, p.y);

            // If we're still in the mask take the border.
            if (pyramid_mask_[scale_iter_].at<uchar>(p.x + 1, p.y) == 0)
            {
                newx.x = 1;
                newx.y = 0;
            }
            else if (pyramid_mask_[scale_iter_].at<uchar>(newx.x + p.x, newx.y + p.y))
                newx.x += 1;
        }
        else
            out = true;
    }

    if (!out)
    {
        pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y) = newx;

        out = false;

        stop = false;
        cost = cost_bullshit(p, curr_cost, stop);

        if (!stop && cost < curr_cost)
        {
            curr_cost = cost;
            old = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y);
        }
        else
            pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y) = old;
    }

    cv::Point newy;
    // If odd, search left.
    if (cpt % 2 != 0)
    {
        if (p.y > 0)
        {
            // Left is better ?
            newy = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y - 1);

            // If we're still in the mask take the border.
            if (pyramid_mask_[scale_iter_].at<uchar>(p.x, p.y - 1) == 0)
            {
                newy.x = 0;
                newy.y = -1;
            }
            else if (pyramid_mask_[scale_iter_].at<uchar>(newy.x + p.x, newy.y + p.y))
                newy.y -= 1;
        }
        else
            out = true;
    }
    // If even, search right.
    else
    {
        if (p.y < pyramid_mask_[scale_iter_].cols - 1)
        {
            // Right is better ?
            newy = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y + 1);

            // If we're still in the mask take the border.
            if (pyramid_mask_[scale_iter_].at<uchar>(p.x, p.y + 1) == 0)
            {
                newy.x = 0;
                newy.y = 1;
            }
            else if (pyramid_mask_[scale_iter_].at<uchar>(newy.x + p.x, newy.y + p.y))
                newy.y += 1;
        }
        else
            out = true;
    }

    if (!out)
    {
        pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y) = newy;

        stop = false;
        cost = cost_bullshit(p, curr_cost, stop);


        if (!stop && cost < curr_cost)
        {
            curr_cost = cost;
            old = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y);
        }
        else
            pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y) = old;
    }

    // Random search.
    random_search(p, curr_cost);
    // Update of the cost.
    pyramid_cost_[scale_iter_].at<float>(p.x, p.y) = curr_cost;
}

void DR::inpaint()
{
  // Compute the cost for every pixels.
  double cost = 0;
  std::list<cv::Point>::iterator it, eit, it2;
  std::list<cv::Point>::reverse_iterator rit, reit;

  // FIXME: scale_iter_ should be passed as argument, use a local variable.
  for (; scale_iter_  >= 0; --scale_iter_)
  {
      std::cout << "SCALE " << scale_iter_ << std::endl;

	  // At the lowest resolution, the init is completely random.
      if (scale_iter_ == max_scale_)
          // Init random.
          build_mask();
      else
      {
          // Random init of the current and merge with the previous one.
          build_mask();
          // Merge.
          offset_scaling_up();
      }

      for (size_t cpt = 0; cpt < iter_; ++cpt)
      {
          if (cpt % 2 != 0)
          {
              it = pyramid_target_pixels_[scale_iter_].begin();
              eit = pyramid_target_pixels_[scale_iter_].end();
              for (; it != eit; ++it)
                  improve(*it, cpt, cost);
          }
          else
          {
              rit = pyramid_target_pixels_[scale_iter_].rbegin();
              reit = pyramid_target_pixels_[scale_iter_].rend();
              for (; rit != reit; ++rit)
                  improve(*rit, cpt, cost);
          }


          cv::Mat temp = pyramid_image_[scale_iter_].clone();
          // Copy the pixel.
          for (it2 = pyramid_target_pixels_[scale_iter_].begin(); it2 != pyramid_target_pixels_[scale_iter_].end(); ++it2)
          {
              cv::Point tmp = pyramid_mapping_[scale_iter_].at<cv::Point>(it2->x, it2->y);
              cv::line(temp, cv::Point(it2->y, it2->x), cv::Point(it2->y + tmp.y, it2->x + tmp.x), cv::Scalar(0, 0, 255));
              patch_copy(*it2, cv::Point(it2->x + tmp.x, it2->y + tmp.y));
          }

          // BEGIN: debug stuff.
          temp = pyramid_image_[scale_iter_].clone();
          //cv::resize(pyramid_image_[scale_iter_], temp, pyramid_image_[0].size(),0, 0, cv::INTER_NEAREST);
          cv::Mat lol = cv::Mat(pyramid_cost_[scale_iter_].size(), CV_8UC1, cv::Scalar(0, 0, 0));

          double minVal;
          double maxVal;
          cv::Point minLoc;
          cv::Point maxLoc;

          cv::minMaxLoc(pyramid_cost_[scale_iter_], &minVal, &maxVal, &minLoc, &maxLoc );
          for (it2 = pyramid_target_pixels_[scale_iter_].begin(); it2 != pyramid_target_pixels_[scale_iter_].end(); ++it2)
          {
              double tmp =  pyramid_cost_[scale_iter_].at<float>(it2->x, it2->y);
              tmp *= 255;
              tmp /= maxVal;

              lol.at<uchar>(it2->x, it2->y) = tmp;
          }

          cv::imshow("scale", temp);
          cv::imshow("cost", lol);
          cv::waitKey(1);
      }
  }
  res_ = pyramid_image_[0].clone();
  cv::imwrite("input/res.jpg", res_);
}
