#include "dr.hh"

int main(int argc, char* argv[])
{

  std::list<std::pair<char*, char*> > tests;
  tests.push_back(std::make_pair(strdup("input/ombres/Guo4.png"), strdup("input/ombres/Guo4.tif")));
  tests.push_back(std::make_pair(strdup("input/ombres/Guo6.png"), strdup("input/ombres/Guo6.tif")));
  tests.push_back(std::make_pair(strdup("input/ombres/Guo1.png"), strdup("input/ombres/Guo1.tif")));
  tests.push_back(std::make_pair(strdup("input/chairi.PNG"), strdup("input/chair.PNG")));
  tests.push_back(std::make_pair(strdup("input/bw1mask.png"), strdup("input/bw1.png")));
  tests.push_back(std::make_pair(strdup("input/her1mask.png"), strdup("input/her1.png")));
//  char* mask = strdup("input/Alexandre/mask_85.png");
//  char* str = strdup("input/Alexandre/0085.jpg");
//  char* mask = strdup("input/panneaumask.png");
//  char* str = strdup("input/panneau.jpg");
//  char* mask = strdup("input/bungeemask.png");
//  char* str = strdup("input/bungeereal.jpg");
//  char* mask = strdup("input/her2mask.png");
//  char* str = strdup("input/her2.png");
//  char* mask = strdup("input/fontainmask.png");
//  char* str = strdup("input/fontainreal.jpg");
//  char* mask = strdup("input/argmask.png");
//  char* str = strdup("input/arg.jpg");
//  char* mask = strdup("input/unnamedmask.png");
//  char* str = strdup("input/unnamed.png");
//  char* mask = strdup("input/ombres/Guo6.png");
//  char* str = strdup("input/ombres/Guo6.tif");
//  char* mask = strdup("input/ombres/Guo1.png");
//  char* str = strdup("input/ombres/Guo1.tif");

  // 2D inpainting, with a binary mask.

  std::list<std::pair<char*, char*> >::iterator it = tests.begin();
  for (; it != tests.end(); ++it)
  {
      DR dr(it->first, it->second/*, DR::INTENSITY*/);
      dr.inpaint();
      cv::waitKey(10000000);
  }

  return 0;
}
