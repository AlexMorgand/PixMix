#include "dr.hh"

bool in_image(unsigned x, unsigned y, cv::Mat& img)
{
	return !(x >= img.rows || x < 0 || y >= img.cols || y < 0);
}

cv::Vec3b get_cost(unsigned x, unsigned y, cv::Mat& img)
{        
	cv::Vec3b res;
	if (!in_image(x, y, img))
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
    cv::Mat tmp(pyramid_mask_[scale_iter_].size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat tmp2 = tmp.clone();
    std::list<cv::Point>::iterator it;
	for (it = pyramid_target_pixels_[scale_iter_].begin(); it != pyramid_target_pixels_[scale_iter_].end(); ++it)
    {
		std::list<cv::Point> fill_list;
		if (in_image(it->x / 2, it->y / 2, pyramid_mapping_[scale_iter_ + 1]))
			fill_list.push_back(cv::Point(it->x / 2 , it->y / 2));
		if (it->x % 2 != 0 && in_image(it->x / 2 + 1, it->y / 2, pyramid_mapping_[scale_iter_ + 1]))
			fill_list.push_back(cv::Point((it->x / 2) + 1 , it->y / 2));
		if (it->y % 2 != 0 && in_image(it->x / 2, it->y / 2 + 1, pyramid_mapping_[scale_iter_ + 1]))
			fill_list.push_back(cv::Point(it->x / 2 , (it->y / 2) + 1));
		if (it->x % 2 != 0 && it->y % 2 != 0 && in_image(it->x / 2 + 1, it->y / 2 + 1, pyramid_mapping_[scale_iter_ + 1]))
			fill_list.push_back(cv::Point((it->x / 2) + 1 , (it->y / 2) + 1));
		auto ab = fill_list.begin();

		bool found = false;
		while (ab != fill_list.end() && !found)
		{
			cv::Point m = pyramid_mapping_[scale_iter_ + 1].at<cv::Point>(ab->x, ab->y);
			m.x *= 2.0;
			m.y *= 2.0;
			found = (pyramid_mask_[scale_iter_ + 1].at<uchar>(ab->x, ab->y)) && (pyramid_mask_[scale_iter_].at<uchar>(it->x + m.x, it->y + m.y) == 0);
			if (found)
				break;
			++ab;
		}

		if (found)
		{
			cv::Point old = pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y);

			// Multiply offset by two due to upsampling.
			cv::Point m = pyramid_mapping_[scale_iter_ + 1].at<cv::Point>(ab->x, ab->y);
			m.x *= 2.0;
			m.y *= 2.0;
			pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y) = m;
			patch_copy(*it, cv::Point(m.x + it->x, m.y + it->y));
			pyramid_cost_[scale_iter_].at<float>(it->x, it->y) = pyramid_cost_[scale_iter_ + 1].at<float>(ab->x, ab->y);
			tmp.at<cv::Vec3b>(it->x, it->y) = pyramid_image_[scale_iter_].at<cv::Vec3b>(m.x + it->x, m.y + it->y);

		}
		else
		{
			//pyramid_cost_[scale_iter_].at<float>(it->x, it->y) = pow(255, 2.0) * 3;
			tmp.at<cv::Vec3b>(it->x, it->y) = cv::Vec3b(255, 0, 0);
		}
		
    }
    cv::resize(tmp, tmp2, pyramid_image_[0].size(), 0, 0, cv::INTER_NEAREST);
    cv::imshow("scaled", tmp2);
}



void DR::init()
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
		pyramid_mapping_[i] =  cv::Mat(curr.size(), CV_32SC2);
	}

	srand(time(0));
	//srand(0);
}

DR::DR(cv::Mat& mask, cv::Mat& input, std::string& prefix, COPY_P cp)
  : cp_(cp),
	// FIXME : Not sure if deepcopy.
    mask_(mask),
    input_(input),
	prefix_(prefix),
    nb_iter_(5),
    iter_(5),
    max_scale_(4),
    scale_iter_(4),
    // PatchMatch : 9, pixmix : 5, good : 11. 15 is phenomenal.
    patch_size_(21),
    pyramid_image_(5),
    pyramid_mapping_(5),
    pyramid_mask_(5),
    pyramid_cost_(5),
    pyramid_size_(5),
    pyramid_target_pixels_(5),
    res_(input_.size(), CV_8UC3)
{
	init();
}


DR::DR(char* mask, char* input, std::string& prefix, COPY_P cp)
	: cp_(cp),
	  prefix_(prefix),
	  mask_(cv::imread(prefix + mask, CV_LOAD_IMAGE_GRAYSCALE)),
	  input_(cv::imread(prefix + input)),
	  nb_iter_(5),
	  iter_(5),
	  max_scale_(4),
	  scale_iter_(4),
	  // PatchMatch : 9, pixmix : 5, good : 11. 15 is phenomenal.
      patch_size_(21),
	  pyramid_image_(5),
	  pyramid_mapping_(5),
	  pyramid_mask_(5),
	  pyramid_cost_(5),
	  pyramid_size_(5),
      pyramid_target_pixels_(5),
	  res_(input_.size(), CV_8UC3)

{
	init();
}


/*double DR::cost_bullshit(cv::Point& p, double curr_cost, bool& stop)
{
	double alpha = 0.5;
	double cost = 0;
	int mini = - (patch_size_ / 2);
	int maxi = patch_size_ / 2;

	cv::Point tmp = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y);

	cv::Mat& tmp_input = pyramid_image_[scale_iter_];

	double d1, d2, d3;

	double spatial = 0;
	std::vector<int> x_vec(4), y_vec(4);
	x_vec[0] = 0; x_vec[1] = -1; x_vec[2] = 0; x_vec[3] = 1;
	y_vec[0] = -1; y_vec[1] = 0; y_vec[2] = 1; y_vec[3] = 0;
	for (unsigned int k = 0; k < 4; ++k)
	//for (int i = mini; i <= maxi && !stop; ++i)
	//{
	//	for (int j = mini; j <= maxi && !stop; ++j)
		{
			int i = x_vec[k]; int j = y_vec[k];
			cv::Vec3b pix_orig = get_cost(p.x + i, p.y + j, tmp_input);
			cv::Vec3b pix_offset = get_cost(p.x + i + tmp.x, p.y + j + tmp.y, tmp_input);
			d1 = pix_orig[0] - pix_offset[0];
			d2 = pix_orig[1] - pix_offset[1];
			d3 = pix_orig[2] - pix_offset[2];
			d1 *= d1;
			d2 *= d2;
			d3 *= d3;
			cost += d1 + d2 + d3;

			// FIXME: min with 200.
			spatial += std::min(pow((p.x + i) - (p.x + i + tmp.x), 2.0) + pow((p.y + j) - (p.y + j + tmp.y), 2.0), 200000.0);// * (1.0/8.0);
			//if (curr_cost <cost)
			if (curr_cost < alpha * cost + (1 - alpha) * spatial)
				stop = true;
		}
	//}

  return alpha * cost + (1 - alpha) * spatial;
  return cost;
}*/

double DR::cost_bullshit(cv::Point& p, double curr_cost, bool& stop)
{
  double cost = 0;
  cv::Point tmp = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y);

  cv::Mat& tmp_input = pyramid_image_[scale_iter_];

  double d1, d2, d3;

  //FIXME: tmp is out ?!
  int mini = - (patch_size_ / 2);
  int maxi = patch_size_ / 2;

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
    cv::waitKey(1);
}

void DR::patch_copy(cv::Point dst, cv::Point src)
{
    if (cp_ == SIMPLE)
        pyramid_image_[scale_iter_].at<cv::Vec3b>(dst.x, dst.y) = pyramid_image_[scale_iter_].at<cv::Vec3b>(src.x, src.y);
    else if (cp_ == INTENSITY)
    {
        int mini = - (patch_size_ / 2);
        int maxi = patch_size_ / 2;

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
        pyramid_image_[scale_iter_].at<cv::Vec3b>(dst.x, dst.y)[1] = g;
        pyramid_image_[scale_iter_].at<cv::Vec3b>(dst.x, dst.y)[2] = b;
    }
}

void DR::random_search(cv::Point& p, double& curr_cost)
{
    //float w = (max_iter_ - scale_iter_ + 1) * 50;
    float w = cv::min(pyramid_mapping_[scale_iter_].rows, pyramid_mapping_[scale_iter_].cols) / 4;
    float alpha = 0.5;

    float alphai;
    bool stop = false;

    // Random value between [-1;1]
    int i = 0;
    float hypo;
    cv::Mat tmp_map = pyramid_mapping_[scale_iter_];

    cv::Point old = tmp_map.at<cv::Point>(p.x, p.y);
    do
    {
		float rx = rand() % 2000 / 1000.f - 1.f;
		float ry = rand() % 2000 / 1000.f - 1.f;
        alphai = pow(alpha, i);
        cv::Point pt(w * alphai * rx, w * alphai * ry);
        hypo = sqrt(pow(pt.x, 2.0) + pow(pt.y, 2.0));

        cv::Point randp(tmp_map.at<cv::Point>(p.x, p.y).x + p.x + pt.x,
                        tmp_map.at<cv::Point>(p.x, p.y).y + p.y + pt.y);

		
        if (in_image(randp.x, randp.y, pyramid_mask_[scale_iter_]) && pyramid_mask_[scale_iter_].at<uchar>(randp.x, randp.y) == 0)
        {
            tmp_map.at<cv::Point>(p.x, p.y).x = randp.x - p.x;
            tmp_map.at<cv::Point>(p.x, p.y).y = randp.y - p.y;

            stop = false;
            double cost = cost_bullshit(p, curr_cost, stop);

            if (!stop && cost < curr_cost)
            {
                curr_cost = cost;
                old = tmp_map.at<cv::Point>(p.x, p.y);
				patch_copy(p, cv::Point(p.x + old.x, p.y + old.y));
            }
            else
                tmp_map.at<cv::Point>(p.x, p.y) = old;
        }

        i++;
    }
    // If below than one pixel break the loop.
    while (hypo >= 1);
}

// FIXME: not sure if we need to fill at the borders !
void DR::improve(cv::Point p, size_t cpt, double& cost)
{
    bool stop;
    bool out = false;
    // Compute cost.
    double curr_cost = pyramid_cost_[scale_iter_].at<float>(p.x, p.y);

    cv::Point old = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y);
    cv::Point newx;
    // If odd, search up.
    if (cpt == 1 || cpt == 3)
    {
		// FIXME: useless conditions.
        if (p.x > 0)
        {
            // Up is better ?
            newx = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x - 1, p.y);
            // Take the pixel above if you're at the border.
            if (in_image(p.x - 1, p.y, pyramid_mask_[scale_iter_]) &&
				pyramid_mask_[scale_iter_].at<uchar>(p.x - 1, p.y) == 0)
            {
                newx.x = -1;
                newx.y = 0;
            }
            else if (in_image(newx.x + p.x, newx.y + p.y, pyramid_mask_[scale_iter_]) &&
				     pyramid_mask_[scale_iter_].at<uchar>(newx.x + p.x, newx.y + p.y))
                newx.x -= 1;
        }
        else
            out = true;
    }
    // If even, search down.
    else if (cpt == 0 || cpt == 4 || cpt == 2)
    {
		// FIXME: useless conditions.
        if (p.x < pyramid_mapping_[scale_iter_].rows - 1)
        {
            // Down is better ?
            newx = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x + 1, p.y);

            // If we're still in the mask take the border.
            if (in_image(p.x + 1, p.y, pyramid_mask_[scale_iter_]) &&
				pyramid_mask_[scale_iter_].at<uchar>(p.x + 1, p.y) == 0)
            {
                newx.x = 1;
                newx.y = 0;
            }
            else if (in_image(newx.x + p.x, newx.y + p.y, pyramid_mask_[scale_iter_]) &&
				     pyramid_mask_[scale_iter_].at<uchar>(newx.x + p.x, newx.y + p.y))
                newx.x += 1;
        }
        else
            out = true;
    }

    if (!out && in_image(p.x + newx.x, p.y + newx.y, pyramid_mapping_[scale_iter_]))
    {
        pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y) = newx;

        out = false;

        stop = false;
        cost = cost_bullshit(p, curr_cost, stop);

        if (!stop && cost < curr_cost)
        {
            curr_cost = cost;
            old = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y);
			patch_copy(p, cv::Point(p.x + old.x, p.y + old.y));

        }
        else
            pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y) = old;
    }

    cv::Point newy;
    // If odd, search left.
    if (cpt == 1 || cpt == 2)
    {
		// FIXME: useless conditions.
        if (p.y > 0)
        {
            // Left is better ?
            newy = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y - 1);

            // If we're still in the mask take the border.
            if (in_image(p.x, p.y - 1, pyramid_mask_[scale_iter_]) &&
				pyramid_mask_[scale_iter_].at<uchar>(p.x, p.y - 1) == 0)
            {
                newy.x = 0;
                newy.y = -1;
            }
            else if (in_image(newy.x + p.x, newy.y + p.y, pyramid_mask_[scale_iter_]) &&
				     pyramid_mask_[scale_iter_].at<uchar>(newy.x + p.x, newy.y + p.y))
                newy.y -= 1;
        }
        else
            out = true;
    }
    // If even, search right.
    else if (cpt == 0 || cpt == 3 || cpt == 4)
    {
		// FIXME: useless conditions.
        if (p.y < pyramid_mask_[scale_iter_].cols - 1)
        {
            // Right is better ?
            newy = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y + 1);

            // If we're still in the mask take the border.
            if (in_image(p.x, p.y + 1, pyramid_mask_[scale_iter_]) &&
				pyramid_mask_[scale_iter_].at<uchar>(p.x, p.y + 1) == 0)
            {
                newy.x = 0;
                newy.y = 1;
            }
            else if (in_image(newy.x + p.x, newy.y + p.y, pyramid_mask_[scale_iter_]) && 
				     pyramid_mask_[scale_iter_].at<uchar>(newy.x + p.x, newy.y + p.y))
                newy.y += 1;
        }
        else
            out = true;
    }

	// FIXME: not pretty 
    if (!out && in_image(p.x + newy.x, p.y + newy.y, pyramid_mapping_[scale_iter_]))
    {
        pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y) = newy;

        stop = false;
        cost = cost_bullshit(p, curr_cost, stop);


        if (!stop && cost < curr_cost)
        {
            curr_cost = cost;
            old = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y);
			patch_copy(p, cv::Point(p.x + old.x, p.y + old.y));
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
	  if (scale_iter_ == 4)
		  patch_size_ = 21;
	  else if (scale_iter_ == 3)
		  patch_size_ = 11;
	  else if (scale_iter_ == 2)
		  patch_size_ = 9;
	  else if (scale_iter_ == 1)
		  patch_size_ = 7;
	  else if (scale_iter_ == 0)
		  patch_size_ = 5;
	  
	  //if (scale_iter_ == 4)
		 // patch_size_ = 25;
	  //else if (scale_iter_ == 3)
		 // patch_size_ = 15;
	  //else if (scale_iter_ == 2)
		 // patch_size_ = 11;
	  //else if (scale_iter_ == 1)
		 // patch_size_ = 9;
	  //else if (scale_iter_ == 0)
		 // patch_size_ = 7;


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
		  unsigned rows = pyramid_mapping_[scale_iter_].rows;
		  unsigned cols = pyramid_mapping_[scale_iter_].cols;
		  if (cpt == 0 || cpt == 4)
		  {
			  for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
				{
					if (pyramid_mask_[scale_iter_].at<uchar>(i, j))
					{
						cv::Point p(i, j);
						improve(p, cpt, cost);
					}
				}
		  }
		  else if (cpt == 1)
		  {
			  for (int i = rows - 1; i >= 0; --i)
				for (int j = cols - 1; j >= 0; --j)
				{
					if (pyramid_mask_[scale_iter_].at<uchar>(i, j))
					{
						cv::Point p(i, j);
						improve(p, cpt, cost);
					}
				}
		  }
		  else if (cpt == 2)
		  {		
			  for (int i = 0; i < rows; ++i)
				for (int j = cols - 1; j >= 0; --j)
				{
					if (pyramid_mask_[scale_iter_].at<uchar>(i, j))
					{
						cv::Point p(i, j);
						improve(p, cpt, cost);
					}
				}
		  }
		  else
		  {
			  for (int i = rows - 1; i >= 0; --i)
				for (int j = 0; j < cols; ++j)
				{
					if (pyramid_mask_[scale_iter_].at<uchar>(i, j))
					{
						cv::Point p(i, j);
						improve(p, cpt, cost);
					}
				}
		  }

          cv::Mat temp = pyramid_image_[scale_iter_].clone();
          
          // BEGIN: debug stuff.
          cv::resize(pyramid_image_[scale_iter_], temp, pyramid_image_[0].size(),0, 0, cv::INTER_NEAREST);
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
  cv::imwrite(prefix_ + std::string("res2.jpg"), res_);
}
