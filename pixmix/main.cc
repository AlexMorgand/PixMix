#include "dr.hh"

int main(int argc, char* argv[])
{
//  char* mask = strdup("input/chairi.PNG");
//  char* str = strdup("input/chair.PNG");
//  char* mask = strdup("input/bw1mask.png");
//  char* str = strdup("input/bw1.png");
//  char* str = strdup("input/her1.png");
//  char* mask = strdup("input/her1mask.png");
  char* mask = strdup("input/Alexandre/mask_85.png");
  char* str = strdup("input/Alexandre/0085.jpg");
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
//  char* mask = strdup("input/ombres/Guo4.png");
//  char* str = strdup("input/ombres/Guo4.tif");
//  char* mask = strdup("input/ombres/Guo1.png");
//  char* str = strdup("input/ombres/Guo1.tif");


  // 2D inpainting, with a binary mask.
  DR dr(mask, str);
  dr.inpaint();
  cv::waitKey(10000000);

  return 0;
}
