#include "dr.hh"

bool in = true;

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
    for (it = pyramid_target_pixels_[scale_iter_ + 1].begin(); it != pyramid_target_pixels_[scale_iter_ + 1].end(); ++it)
    {
        cv::Point ab(it->x * 2, it->y * 2);
        // FIXME: not sure.
       /* if (a >= pyramid_mapping_[scale_iter_].rows - 1)
            a = pyramid_mapping_[scale_iter_].rows - 2;*/

        /*if (b >= pyramid_mapping_[scale_iter_].cols - 1)
            b = pyramid_mapping_[scale_iter_].cols -2;*/

        cv::Point old = pyramid_mapping_[scale_iter_].at<cv::Point>(ab.x, ab.y);

        // Multiply offset by two due to upsampling.
        cv::Point m = pyramid_mapping_[scale_iter_ + 1].at<cv::Point>(it->x, it->y);
        m.x *= 2.0;
        m.y *= 2.0;
        pyramid_mapping_[scale_iter_].at<cv::Point>(ab.x, ab.y) = m;

        //cost = alpha * spatial_cost(*it) + (1.0 - alpha) * appearance_cost(*it);
        double cost = cost_bullshit(ab);
            std::cout << "WHAT " << (pyramid_cost_[scale_iter_ + 1].at<float>(it->x, it->y)) << " " << cost << std::endl;

        // Random is better than the previous guess in the small scale.
        if (pyramid_cost_[scale_iter_].at<float>(it->x, it->y) < cost)
        {
            std::cout << "RANDOM is better\n";
            pyramid_mapping_[scale_iter_].at<cv::Point>(ab.x, ab.y) = old;
        }
        else
        {
            pyramid_image_[scale_iter_].at<cv::Vec3b>(ab.x, ab.y) = pyramid_image_[scale_iter_].at<cv::Vec3b>(m.x + ab.x, m.y + ab.y);
            pyramid_cost_[scale_iter_].at<float>(ab.x, ab.y) = pyramid_cost_[scale_iter_ + 1].at<float>(it->x, it->y);
            tmp.at<cv::Vec3b>(ab.x, ab.y) = pyramid_image_[scale_iter_].at<cv::Vec3b>(m.x + ab.x, m.y + ab.y);
        }

    }
    cv::resize(tmp, tmp2, pyramid_image_[0].size(), 0, 0, cv::INTER_NEAREST);
    cv::imshow("scaled", tmp2);
    cv::waitKey(1000000);
}

DR::DR(char* mask, char* input)
  : mask_(cv::imread(mask, CV_LOAD_IMAGE_GRAYSCALE)),
    input_(cv::imread(input)),
    gray_(mask_.size(), CV_8UC1),
    odd(true),
    nb_iter_(5),
    iter_(5),
    max_iter_(5),
    max_scale_(5),
    scale_iter_(5),
    patch_size_(5),
    pyramid_image_(6),
    pyramid_mapping_(6),
    pyramid_mask_(6),
    pyramid_cost_(6),
    pyramid_target_pixels_(6)
{
  gray_ = cv::Mat(mask_.size(), CV_8UC1);
  cv::cvtColor(input_, gray_, CV_BGR2GRAY);

  cv::Mat curr = input_;
  cv::Mat curr_m = mask_;
  pyramid_image_[0] = input_;
  pyramid_cost_[0] = cv::Mat(curr.size(), CV_32FC1);
  pyramid_mapping_[0] = cv::Mat(curr.size(), CV_32SC2);
  pyramid_mask_[0] = mask_;

  for (size_t i = 1; i <= max_scale_; ++i)
  {
      curr = Scale(1, curr);
      pyramid_image_[i] = curr;

      curr_m = Scale(1, curr_m);
      pyramid_mask_[i] = curr_m;

      pyramid_cost_[i] = cv::Mat(curr.size(), CV_32FC1);
      pyramid_mapping_[i] = cv::Mat(curr.size(), CV_32SC2);
  }

  srand(time(0));
}

// FIXME: early termination TODO.
double DR::cost_bullshit(cv::Point& p)
{
  double cost = 0;
  cv::Point tmp = pyramid_mapping_[scale_iter_].at<cv::Point>(p.x, p.y);

  cv::Mat tmp_input = pyramid_image_[scale_iter_];
  double d1 = 0, d2 = 0, d3 = 0;

  for (int i = - patch_size_ / 2; i <= patch_size_ / 2; ++i)
  {
    for (int j = - patch_size_ / 2; j <= patch_size_ / 2; ++j)
    {
        d1 = 0, d2 = 0, d3 = 0;

        d1 =  tmp_input.at<cv::Vec3b>(p.x + i, p.y + j)[0] -  tmp_input.at<cv::Vec3b>(p.x + tmp.x + i, p.y + tmp.y + j)[0];
        d1 *= d1;
        d2 = tmp_input.at<cv::Vec3b>(p.x + i, p.y + j)[1] - tmp_input.at<cv::Vec3b>(p.x + tmp.x + i, p.y + tmp.y + j)[1];
        d2 *= d2;
        d3 =  tmp_input.at<cv::Vec3b>(p.x + i, p.y + j)[2] - tmp_input.at<cv::Vec3b>(p.x + tmp.x + i, p.y + tmp.y + j)[2];
        d3 *= d3;
        cost += d1 + d2 + d3;

        /*if (!in && i == 0 && j == 0)
        {
       cv::Mat t = input_.clone();
        cv::line(t, cv::Point(p.y +j, p.x + i), cv::Point(p.y + tmp.y + j, p.x + tmp.x + i), cv::Scalar(0, 0, 255));
        cv::imshow("Line", t);
        std::cout << " offset " << tmp.x << " " << tmp.y << std::endl;

        std::cout << "d1,2,3 " << d1 << " " << d2 << " " << d3 << std::endl;
       cv::waitKey(1000);
        }*/

    }
  }
  return cost;

}

void DR::build_mask()
{
    size_t rows = pyramid_mask_[scale_iter_].rows;
    size_t cols = pyramid_mask_[scale_iter_].cols;

    int x, y;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (pyramid_mask_[scale_iter_].at<uchar>(i, j))
            {
                pyramid_target_pixels_[scale_iter_].push_back(cv::Point(i, j));
                do
                {
                    x = rand() % rows;
                    y = rand() % cols;
                } while (pyramid_mask_[scale_iter_].at<uchar>(x, y));

                cv::Point m = cv::Point(x - i, y - j);
                pyramid_mapping_[scale_iter_].at<cv::Point>(i, j) = m;

                pyramid_image_[scale_iter_].at<cv::Vec3b>(i, j) = pyramid_image_[scale_iter_].at<cv::Vec3b>(m.x + i, m.y + j);

            }
            else
                pyramid_mapping_[scale_iter_].at<cv::Point>(i, j) = cv::Point(0, 0);
        }
    }

    double alpha = 0.5;
    std::list<cv::Point>::iterator it;
    for (it = pyramid_target_pixels_[scale_iter_].begin(); it != pyramid_target_pixels_[scale_iter_].end(); ++it)
    {
        pyramid_cost_[scale_iter_].at<float>(it->x, it->y) = cost_bullshit(*it);
        //cost_.at<float>(it->x, it->y) = alpha * spatial_cost(*it) + (1.0 - alpha) * appearance_cost(*it);
    }
    in = false;

}

void DR::random_search(cv::Point& p, double& curr_cost)
{
    // FIXME: instead of 50 => min(rows, cols) / 2 ?
    double w = (max_iter_ - scale_iter_ + 1) * 50;
    double alpha = 0.5;

    // Random value between [-1;1]
    float rx = rand() % 2000 / 1000.f - 1.f;
    float ry = rand() % 2000 / 1000.f - 1.f;
    int i = 0;
    float hypo;
    cv::Mat tmp_map = pyramid_mapping_[scale_iter_];

    cv::Point old = tmp_map.at<cv::Point>(p.x, p.y);
    do
    {
        cv::Point pt(w * pow(alpha, i) * rx, w * pow(alpha, i) * ry);
        hypo = sqrt(pow(pt.x, 2) + pow(pt.y, 2));

        cv::Point randp(tmp_map.at<cv::Point>(p.x, p.y).x + p.x + pt.x,
                        tmp_map.at<cv::Point>(p.x, p.y).y + p.y + pt.y);

        if ((randp.x >= 0 && randp.x <= tmp_map.cols -1) &&
             (randp.y >= 0 && randp.y <= tmp_map.rows-1))
        {

            if (pyramid_mask_[scale_iter_].at<uchar>(randp.x, randp.y) == 0)
            {
                tmp_map.at<cv::Point>(p.x, p.y).x = randp.x - p.x;
                tmp_map.at<cv::Point>(p.x, p.y).y = randp.y - p.y;

                /*cv::Mat t = pyramid_image_[scale_iter_].clone();
                cv::line(t, cv::Point(p.y, p.x), cv::Point(randp.y, randp.x), cv::Scalar(1, 0, 255));
                cv::imshow("Line", t);
                cv::waitKey(10000000);*/



                //double cost = alpha * spatial_cost(p) + (1.0 - alpha) * appearance_cost(p);
                double cost = cost_bullshit(p);
                //std::cout << "cost rand " << cost << std::endl;

                if (cost < curr_cost)
                {
                    curr_cost = cost;
                    old = tmp_map.at<cv::Point>(p.x, p.y);
                    std::cout << "CHOOSE RANDOM\n";
                }
                else
                {
                    tmp_map.at<cv::Point>(p.x, p.y) = old;
                }
            }
        }
        i++;
    }
    // If below than one pixel break the loop.
    while (hypo >= 1);
    //std::cout << " ==========================\n";
}

//FIXME: use a cost array to avoid recomputing the current cost.
void DR::inpaint()
{
  // Compute the cost for every pixels.
  double alpha = 0.5;
  // FIXME: Find the min.
  double cost = 0;
  std::list<cv::Point>::iterator it, it2;

  for (; scale_iter_ >= 0; --scale_iter_)
  {
      std::cout << scale_iter_ << std::endl;
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
          for (it = pyramid_target_pixels_[scale_iter_].begin(); it != pyramid_target_pixels_[scale_iter_].end(); ++it)
          {
              // Compute cost.
              double curr_cost = pyramid_cost_[scale_iter_].at<float>(it->x, it->y);
              // double curr_cost =
              //   alpha * spatial_cost(*it) + (1.0 - alpha) * appearance_cost(*it);

          //    std::cout << "point " << it->x << " " << it->y << std::endl;
              // Up is better ?

              cv::Point old = pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y);
              cv::Point newx;
              if (cpt % 2 != 0)
              {
                  newx = pyramid_mapping_[scale_iter_].at<cv::Point>(it->x - 1, it->y);
                  // FIXME: compare with the neighbor: To erase !
                  newx.x = (newx.x + 1);

                  // If we're still in the mask take the border.
                  if (pyramid_mask_[scale_iter_].at<uchar>(it->x - 1, it->y) == 0)
                  {
                      newx.x = -1;
                      newx.y = 0;
                  }
                  else if (pyramid_mask_[scale_iter_].at<uchar>(newx.x + it->x, newx.y + it->y))
                      newx.x -= 2;
              }
              else
              {
                  newx = pyramid_mapping_[scale_iter_].at<cv::Point>(it->x + 1, it->y);
                  // FIXME: compare with the neighbor: To erase !
                  newx.x = (newx.x - 1);

                  // If we're still in the mask take the border.
                  if (pyramid_mask_[scale_iter_].at<uchar>(it->x + 1, it->y) == 0)
                  {
                      newx.x = 1;
                      newx.y = 0;
                  }
                  else if (pyramid_mask_[scale_iter_].at<uchar>(newx.x + it->x, newx.y + it->y))
                      newx.x += 2;
              }

              // FIXME: UGLY.
              if (newx.x + it->x >= input_.rows)
                  newx.x = input_.rows - 1 - it->x;

              pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y) = newx;

              //cost = alpha * spatial_cost(*it) + (1.0 - alpha) * appearance_cost(*it);
              cost = cost_bullshit(*it);

            //  std::cout << "curr cost " << curr_cost << std::endl;
            //  std::cout << "UP : costx " << cost << std::endl;

              if (cost < curr_cost)
              {
                  curr_cost = cost;
                  old = pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y);
              //    std::cout << "CHOOSE UP\n";
              }
              else
                  pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y) = old;

              // Left is better ?
              cv::Point newy;
              if (cpt % 2 != 0)
              {
                  newy = pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y - 1);
                  // FIXME: compare with the neighbor: To erase !
                  newy.y = newy.y + 1;

                  // If we're still in the mask take the border.

                  if (pyramid_mask_[scale_iter_].at<uchar>(it->x, it->y - 1) == 0)
                  {
                      newy.x = 0;
                      newy.y = -1;
                  }
                  else if (pyramid_mask_[scale_iter_].at<uchar>(newy.x + it->x, newy.y + it->y))
                      newy.y -= 2;
              }
              else
              {
                  newy = pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y + 1);
                  // FIXME: compare with the neighbor: To erase !
                  newy.y = newy.y - 1;
                  // If we're still in the mask take the border.

                  if (pyramid_mask_[scale_iter_].at<uchar>(it->x, it->y + 1) == 0)
                  {
                      newy.x = 0;
                      newy.y = 1;
                  }
                  else if (pyramid_mask_[scale_iter_].at<uchar>(newy.x + it->x, newy.y + it->y))
                      newy.y += 2;
              }
              // FIXME: UGLY.
              if (newy.y + it->y >= input_.cols)
                  newy.y = input_.cols - 1 - it->y;

              pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y) = newy;

              //cost = alpha * spatial_cost(*it) + (1.0 - alpha) * appearance_cost(*it);
              cost = cost_bullshit(*it);

             // std::cout << "LEFT : costy " << cost << std::endl;

              if (cost < curr_cost)
              {
                  curr_cost = cost;
                  old = pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y);
               //   std::cout << "CHOOSE LEFT\n";
              }
              else
                  pyramid_mapping_[scale_iter_].at<cv::Point>(it->x, it->y) = old;

              in = true;
              // Random search.
              random_search(*it, curr_cost);
              // Update of the cost.
              pyramid_cost_[scale_iter_].at<float>(it->x, it->y) = curr_cost;
              in = false;
          }

          // FIXME: segf.
          for (it2 = pyramid_target_pixels_[scale_iter_].begin(); it2 != pyramid_target_pixels_[scale_iter_].end(); ++it2)
          {
              cv::Mat temp = pyramid_image_[scale_iter_].clone();

              cv::Point tmp = pyramid_mapping_[scale_iter_].at<cv::Point>(it2->x, it2->y);

          //    cv::line(temp, cv::Point(it2->y, it2->x), cv::Point(it2->y + tmp.y, it2->x + tmp.x), cv::Scalar(0, 0, 255));

              pyramid_image_[scale_iter_].at<cv::Vec3b>(it2->x, it2->y) = pyramid_image_[scale_iter_].at<cv::Vec3b>(it2->x + tmp.x, it2->y + tmp.y);

          }
          cv::Mat temp = pyramid_image_[scale_iter_].clone();
          cv::resize(pyramid_image_[scale_iter_], temp, pyramid_image_[0].size(),0, 0, cv::INTER_NEAREST);
          cv::imshow("scale", temp);
          cv::waitKey(10000);
      }
  }

}
/*
double DR::spatial_cost(cv::Point& p)
{
  // FIXME: stop the cost if it's bigger than the current cost.
  // We fixed a 8 neighbor regions.
  double cost = 0;
  for (int i = -2; i <= 2; ++i)
  {
    for (int j = -2; j <= 2; ++j)
    {
      if (j != 0 || i != 0)
      {
        // Sum of squared differences.
        // f(p) + v.
        cv::Point tmp = mapping_.at<cv::Point>(p.x, p.y);
        tmp.x += i;
        tmp.y += j;
        cv::Point SSD = tmp - mapping_.at<cv::Point>(p.x + i, p.y + j);
        // ds => min(|p0 - p1|², distance max).
        double cost_tmp = SSD.x * SSD.x + SSD.y * SSD.y;
        // FIXME: check the 200.
        cost_tmp = std::min(cost_tmp, 200.0) / 24;
        cost += cost_tmp;
      }
    }
  }
  return cost;
}

// FIXME: check the herling implementation, not sure.
double DR::weightning(cv::Point& p)
{
  double cost = 0;
  for (int i = -2; i <= 2; ++i)
  {
    for (int j = -2; j <= 2; ++j)
    {
      if (j != 0 || i != 0)
      {
        cv::Point tmp = mapping_.at<cv::Point>(p.x + i, p.y + j);
        double diff = (int) gray_.at<uchar>(p.x + i, p.y + j) -
                      (int) gray_.at<uchar>(tmp.x, tmp.y);

//        std::cout << "gray " << (int) gray_.at<uchar>(p.x + i, p.y + j) << " " << (int) gray_.at<uchar>(tmp.x, tmp.y) << std::endl;
        cost += diff * diff;
  //      std::cout << "evolution " << cost  << " with diff " << diff << " \n";
      }
    }
  }
//  std::cout << cost << " \n";
//  std::cout << exp(-sqrt(cost)) << std::endl;
  //cv::waitKey(1000);
  return exp(-sqrt(cost));
}

double DR::appearance_cost(cv::Point& p)
{
  // FIXME: stop the cost if it's bigger than the current cost.
  // We fixed a 8 neighbor regions.
  double cost = 0;
  for (int i = -2; i <= 2; ++i)
  {
    for (int j = -2; j <= 2; ++j)
    {
      if (j != 0 || i != 0)
      {
        // Sum of squared differences.
        // f(p) + v.
        cv::Point tmp = p;
        tmp.x += i;
        tmp.y += j;
        cv::Point tmp2 = mapping_.at<cv::Point>(p.x, p.y);
        tmp2.x += i;
        tmp2.y += j;
        cv::Vec3b diff =
          input_.at<cv::Vec3b>(i + tmp.x, j + tmp.y) - input_.at<cv::Vec3b>(i + tmp2.x, j + tmp2.y);
        cost += diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
        //int diff = gray_.at<uchar>(tmp.x, tmp.y) - gray_.at<uchar>(tmp2.x, tmp2.y);
        //cost += diff * diff;
        // FIXME: find the good ponderation.
      //  cost *= weightning(tmp);
      }
    }
  }
  return cost;
}*/
