// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <rmd/publisher.h>

#include <rmd/seed_matrix.cuh>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Eigen>


rmd::Publisher::Publisher(ros::NodeHandle &nh,
                          std::shared_ptr<rmd::Depthmap> depthmap)
  : nh_(nh)
  , pc_(new PointCloud)
  , pcc_(new PointCloudColor)
{
  depthmap_ = depthmap;
  colored_.create(depthmap->getHeight(), depthmap_->getWidth(), CV_8UC3);
  image_transport::ImageTransport it(nh_);
  depthmap_publisher_ = it.advertise("remode/depth",       10);
  conv_publisher_     = it.advertise("remode/convergence", 10);
  pub_pcc_ = nh_.advertise<PointCloudColor>("remode/pointcloudcolored", 1);

}

void rmd::Publisher::publishDepthmap() const
{
  cv_bridge::CvImage cv_image;
  cv_image.header.frame_id = "depthmap";
  cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  cv_image.image = depthmap_->getDepthmap();
  if(nh_.ok())
  {
    cv_image.header.stamp = ros::Time::now();
    depthmap_publisher_.publish(cv_image.toImageMsg());
    std::cout << "INFO: publishing depth map" << std::endl;
  }
}

void rmd::Publisher::publishPointCloud() const
{
  {
    std::lock_guard<std::mutex> lock(depthmap_->getRefImgMutex());

    const cv::Mat depth = depthmap_->getDepthmap();
    const cv::Mat convergence = depthmap_->getConvergenceMap();
    const cv::Mat ref_img_color = depthmap_->getReferenceColorImage();
    const rmd::SE3<float> T_world_ref = depthmap_->getT_world_ref();
    
    cv_bridge::CvImage img_bgr_msg;    
    img_bgr_msg.header.frame_id = "/cam_mono";
    img_bgr_msg.image = ref_img_color;
    img_bgr_msg.encoding = sensor_msgs::image_encodings::BGR8;

    pub_img_bgr_.publish(img_bgr_msg.toImageMsg());
    const float fx = depthmap_->getFx();
    const float fy = depthmap_->getFy();
    const float cx = depthmap_->getCx();
    const float cy = depthmap_->getCy();

    for(int y=0; y<depth.rows; ++y)
    {
      for(int x=0; x<depth.cols; ++x)
      {
        const float3 f = normalize( make_float3((x-cx)/fx, (y-cy)/fy, 1.0f) );
        const float3 xyz = T_world_ref * ( f * depth.at<float>(y, x) );
        if( rmd::ConvergenceState::CONVERGED == convergence.at<int>(y, x) )
        {
          PointTypeColor cp;
          cp.x = xyz.x;
          cp.y = xyz.y;
          cp.z = xyz.z;
          cv::Vec3i intensity_rgb = ref_img_color.at<cv::Vec3i>(y, x);
          uint8_t b = (uint8_t) (ref_img_color.data[ref_img_color.channels()*(ref_img_color.cols*y + x) + 0]);    
          uint8_t g = (uint8_t) (ref_img_color.data[ref_img_color.channels()*(ref_img_color.cols*y + x) + 1]);
          uint8_t r = (uint8_t) (ref_img_color.data[ref_img_color.channels()*(ref_img_color.cols*y + x) + 2]);
          uint32_t rgb_c = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
          cp.rgb = *reinterpret_cast<float*>(&rgb_c);
          pcc_->push_back(cp);
        }
      }
    }
  }
  if (!pcc_->empty())
  {
    if(nh_.ok())
    {
      uint64_t timestamp;
#if PCL_MAJOR_VERSION == 1 && PCL_MINOR_VERSION >= 7
      pcl_conversions::toPCL(ros::Time::now(), timestamp);
#else
      timestamp = ros::Time::now();
#endif
      pcc_->header.frame_id = "/world";
      pcc_->header.stamp = timestamp;
      pub_pcc_.publish(pcc_);
      std::cout << "INFO: publishing pointcloud, " << pc_->size() << " points" << std::endl;
    }
  }
}

void rmd::Publisher::publishDepthmapAndPointCloud() const
{
  publishDepthmap();
  publishPointCloud();
}

void rmd::Publisher::publishConvergenceMap()
{
  std::lock_guard<std::mutex> lock(depthmap_->getRefImgMutex());

  const cv::Mat convergence = depthmap_->getConvergenceMap();
  const cv::Mat ref_img = depthmap_->getReferenceImage();

  cv::cvtColor(ref_img, colored_, CV_GRAY2BGR);
  for (int r = 0; r < colored_.rows; r++)
  {
    for (int c = 0; c < colored_.cols; c++)
    {
      switch(convergence.at<int>(r, c))
      {
      case rmd::ConvergenceState::CONVERGED:
        colored_.at<cv::Vec3b>(r, c)[0] = 255;
        break;
      case rmd::ConvergenceState::DIVERGED:
        colored_.at<cv::Vec3b>(r, c)[2] = 255;
        break;
      default:
        break;
      }
    }
  }
  cv_bridge::CvImage cv_image;
  cv_image.header.frame_id = "convergence_map";
  cv_image.encoding = sensor_msgs::image_encodings::BGR8;
  cv_image.image = colored_;
  if(nh_.ok())
  {
    cv_image.header.stamp = ros::Time::now();
    conv_publisher_.publish(cv_image.toImageMsg());
    std::cout << "INFO: publishing convergence map" << std::endl;
  }
}
