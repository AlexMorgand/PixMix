#include "dr.hh"

int main(int argc, char* argv[])
{
	std::string prefix("C:/Users/am237982/Desktop/cea/CEA/Alexandre/Dev/supermedia_inpaint/PixMix/pixmix/input/");

  std::list<std::pair<char*, char*> > tests;
  //tests.push_back(std::make_pair(strdup("ombres/Guo4.png"), strdup("ombres/Guo4.tif")));
  //tests.push_back(std::make_pair(strdup("ombres/Guo6.png"), strdup("ombres/Guo6.tif")));
  //tests.push_back(std::make_pair(strdup("ombres/Guo1.png"), strdup("ombres/Guo1.tif")));
  //tests.push_back(std::make_pair(strdup("chairi.PNG"), strdup("chair.PNG")));
  //tests.push_back(std::make_pair(strdup("bw1mask.png"), strdup("bw1.png")));
  //tests.push_back(std::make_pair(strdup("her2mask.png"), strdup("her2.png")));
  //tests.push_back(std::make_pair(strdup("her1mask.png"), strdup("her1.png")));
  //tests.push_back(std::make_pair(strdup("argmask.png"), strdup("arg.jpg")));
  tests.push_back(std::make_pair(strdup("1mask.png"), strdup("1.png")));
  //tests.push_back(std::make_pair(strdup("2mask.png"), strdup("2.png")));
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
      DR dr(it->first, it->second, prefix/*, DR::INTENSITY*/);
      dr.inpaint();
      cv::waitKey(0);
  }

  return 0;
}
