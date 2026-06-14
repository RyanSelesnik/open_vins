/*
 * OpenVINS sim driver — *live* variant
 *
 * Same VIO pipeline as `run_simulation`, but instead of the internal
 * BsplineSE3 trajectory + IMU model driving the filter, we subscribe to
 * an external IMU stream and an external ground-truth odometry topic.
 *
 * For every odom callback we re-project the loaded feature map into the
 * virtual camera at that pose and feed the resulting (id, uv) pairs into
 * the MSCKF as if a tracker had produced them. Both the uv measurements and
 * the IMU samples are corrupted with the Simulator's own noise model
 * (pixel noise, IMU white noise + random-walk biases), since LA-Planner's
 * sim publishes noiseless ideal values.
 *
 * Topics (ROS 1):
 *   ~imu_topic         (default "/quadrotor_simulator_so3/imu"  sensor_msgs/Imu)
 *   ~odom_topic        (default "/state_ukf/odom"               nav_msgs/Odometry)
 *   ~cam_rate_hz       (default 10.0)   sub-sample odom to this rate for camera feed
 *
 * Featmap path comes from `sim_feature_map_path` in the estimator config,
 * exactly as in run_simulation.
 */

#include <csignal>
#include <memory>

#include "core/VioManager.h"
#include "sim/Simulator.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "ros/ROS1Visualizer.h"
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/Odometry.h>

using namespace ov_msckf;

namespace {

std::shared_ptr<Simulator> sim;
std::shared_ptr<VioManager> sys;
std::shared_ptr<ROS1Visualizer> viz;

bool filter_initialized = false;
double last_cam_time = -1.0;
double cam_period = 0.1;  // 10 Hz default

void imu_callback(const sensor_msgs::Imu::ConstPtr &msg) {
  ov_core::ImuData m;
  // LA-Planner's quadrotor_simulator_so3 publishes IMU with header.stamp=0.
  // Substitute wall-clock so OV's IMU propagator gets monotonic timestamps.
  m.timestamp = msg->header.stamp.toSec();
  if (m.timestamp <= 0.0) m.timestamp = ros::Time::now().toSec();
  m.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  // LA-Planner's sim fills linear_acceleration with the TRUE world-frame
  // acceleration (gravity already cancelled: acc_ = v_dot in Quadrotor.cpp).
  // A real accelerometer measures body-frame specific force R_GtoI*(a_w + g).
  // Reconstruct it using the attitude the sim ships in the same message,
  // otherwise the propagator free-falls at -g.
  Eigen::Quaterniond q_ItoG(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
  Eigen::Vector3d a_world(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  Eigen::Vector3d gravity(0.0, 0.0, 9.81);  // matches sim g_ and gravity_mag in config
  m.am = q_ItoG.toRotationMatrix().transpose() * (a_world + gravity);
  if (!filter_initialized) return;  // ignore IMU until we have a pose to init from
  static double last_imu_time = -1.0;
  double dt = (last_imu_time > 0) ? (m.timestamp - last_imu_time) : 0.005;
  last_imu_time = m.timestamp;
  if (dt <= 0) return;  // drop out-of-order/duplicate stamps
  sim->perturb_imu_measurement(m.timestamp, dt, m.wm, m.am);
  sys->feed_measurement_imu(m);
  viz->visualize_odometry(m.timestamp);
  static int n_imu = 0;
  if (++n_imu % 200 == 0) {
    PRINT_INFO(CYAN "[SUB-SIM]: imu n=%d t=%.3f am=[%.2f %.2f %.2f] wm=[%.2f %.2f %.2f]\n" RESET,
               n_imu, m.timestamp, m.am.x(), m.am.y(), m.am.z(), m.wm.x(), m.wm.y(), m.wm.z());
  }
}

void odom_callback(const nav_msgs::Odometry::ConstPtr &msg) {
  double t = msg->header.stamp.toSec();

  // Pose: world -> body (FLU). msg gives p_IinG + q_ItoG (Hamilton quat).
  Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                       msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
  Eigen::Matrix3d R_ItoG = q.toRotationMatrix();
  Eigen::Matrix3d R_GtoI = R_ItoG.transpose();
  Eigen::Vector3d p_IinG(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);

  // Initialize filter from the first odom message of a SETTLED quad.
  // Initializing mid-takeoff (fast node startup can race the takeoff
  // transient) drops a fresh filter -- no clones, zero bias knowledge --
  // straight into an aggressive-acceleration regime and it diverges within
  // seconds. Require ~1 s of near-zero velocity first; both the on-ground
  // wait and the post-takeoff hover satisfy it.
  if (!filter_initialized) {
    Eigen::Vector3d v_init(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
    static int n_still = 0;
    n_still = (v_init.norm() < 0.05) ? n_still + 1 : 0;
    if (n_still < 200) return;  // 1 s at the 200 Hz odom rate
    Eigen::Matrix<double, 17, 1> imustate = Eigen::Matrix<double, 17, 1>::Zero();
    imustate(0, 0) = t - sim->get_true_parameters().calib_camimu_dt;
    // JPL quaternion convention: [qx, qy, qz, qw] for q_GtoI
    Eigen::Quaterniond q_GtoI(R_GtoI);
    imustate(1, 0) = q_GtoI.x();
    imustate(2, 0) = q_GtoI.y();
    imustate(3, 0) = q_GtoI.z();
    imustate(4, 0) = q_GtoI.w();
    imustate(5, 0) = p_IinG.x();
    imustate(6, 0) = p_IinG.y();
    imustate(7, 0) = p_IinG.z();
    // velocity (msg twist if available, else zero)
    imustate(8, 0)  = msg->twist.twist.linear.x;
    imustate(9, 0)  = msg->twist.twist.linear.y;
    imustate(10, 0) = msg->twist.twist.linear.z;
    sys->initialize_with_gt(imustate);
    filter_initialized = true;
    PRINT_INFO(GREEN "[SUB-SIM]: filter initialized at t=%.3f, p=[%.2f %.2f %.2f]\n" RESET,
               t, p_IinG.x(), p_IinG.y(), p_IinG.z());
    return;
  }

  // Throttle camera feed
  if (last_cam_time > 0 && (t - last_cam_time) < cam_period) return;
  last_cam_time = t;

  // Project featmap into cam 0 at this pose.
  auto feats = sim->project_pointcloud(R_GtoI, p_IinG, 0, sim->get_map());
  static int n_cam = 0;
  ++n_cam;
  if (n_cam <= 5 || n_cam % 50 == 0) {
    PRINT_INFO(MAGENTA "[SUB-SIM]: cam n=%d t=%.3f p=[%.2f %.2f %.2f] feats_in_fov=%zu / %zu\n" RESET,
               n_cam, t, p_IinG.x(), p_IinG.y(), p_IinG.z(), feats.size(), sim->get_map().size());
  }
  if (feats.empty()) {
    return;
  }

  // Mirror get_next_cam: cap at num_pts (tracker budget), then add pixel noise
  if ((int)feats.size() > sim->get_true_parameters().num_pts) {
    feats.erase(feats.begin() + sim->get_true_parameters().num_pts, feats.end());
  }
  sim->perturb_camera_measurements(0, feats);

  std::vector<int> camids = {0};
  std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> all_feats = {feats};
  sys->feed_measurement_simulation(t, camids, all_feats);
  viz->visualize();
}

void featmap_callback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  // Convert sensor_msgs/PointCloud2 -> std::vector<Vector3d>, hand to Simulator.
  // The planner publishes /feature/feature_map as a cumulative cloud of
  // every 3D feature it has observed so far (line-of-sight + FOV checked
  // by LA-Planner side). We replace the bridge's internal featmap with this
  // each time the cloud updates. set_featmap_from_points preserves IDs for
  // points within eps of existing entries so MSCKF tracks stay coherent.
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(msg->width * msg->height);
  sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iy(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iz(*msg, "z");
  for (; ix != ix.end(); ++ix, ++iy, ++iz) {
    pts.emplace_back(*ix, *iy, *iz);
  }
  if (pts.empty()) return;
  sim->set_featmap_from_points(pts);
  static int n = 0;
  if (++n == 1 || n % 50 == 0) {
    PRINT_INFO(CYAN "[SUB-SIM]: featmap update #%d: %zu pts in, %zu pts internal\n" RESET,
               n, pts.size(), sim->get_map().size());
  }
}

void signal_callback_handler(int /*signum*/) { ros::shutdown(); }

}  // namespace

int main(int argc, char **argv) {
  ros::init(argc, argv, "run_subscribe_simulation");
  auto nh = std::make_shared<ros::NodeHandle>("~");

  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) config_path = argv[1];
  nh->param<std::string>("config_path", config_path, config_path);

  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  parser->set_node_handler(nh);

  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);
  params.print_and_load_simulation(parser);
  params.num_opencv_threads = 0;
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;
  sim = std::make_shared<Simulator>(params);
  sys = std::make_shared<VioManager>(params);
  viz = std::make_shared<ROS1Visualizer>(nh, sys, sim);

  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Topic params + rate
  std::string imu_topic, odom_topic, featmap_topic;
  double cam_rate_hz = 10.0;
  bool featmap_subscribe = false;
  nh->param<std::string>("imu_topic",       imu_topic,       "/quadrotor_simulator_so3/imu");
  nh->param<std::string>("odom_topic",      odom_topic,      "/state_ukf/odom");
  nh->param<std::string>("featmap_topic",   featmap_topic,   "/feature/feature_map");
  nh->param<bool>("featmap_subscribe",      featmap_subscribe, false);
  nh->param<double>("cam_rate_hz",          cam_rate_hz,     10.0);
  cam_period = 1.0 / cam_rate_hz;

  // If we're subscribing for the featmap, wipe the constructor's random map
  // (the random ones live inside the spline's bounding region, unrelated to
  // LA-Planner's scene). The cloud subscriber will populate from /feature/feature_map.
  if (featmap_subscribe) {
    sim->clear_map();
    PRINT_INFO(YELLOW "[SUB-SIM]: featmap mode = LIVE SUBSCRIBE (topic %s) — internal map cleared\n" RESET,
               featmap_topic.c_str());
  }

  PRINT_INFO(GREEN "[SUB-SIM]: imu_topic=%s odom_topic=%s cam_rate=%.1fHz featmap=%lu pts (mode=%s)\n" RESET,
             imu_topic.c_str(), odom_topic.c_str(), cam_rate_hz, sim->get_map().size(),
             featmap_subscribe ? "subscribe" : "static");

  signal(SIGINT, signal_callback_handler);

  ros::Subscriber sub_imu  = nh->subscribe(imu_topic,  500, imu_callback);
  ros::Subscriber sub_odom = nh->subscribe(odom_topic, 100, odom_callback);
  ros::Subscriber sub_fmap;
  if (featmap_subscribe) {
    sub_fmap = nh->subscribe(featmap_topic, 5, featmap_callback);
  }

  ros::spin();

  viz->visualize_final();
  ros::shutdown();
  return EXIT_SUCCESS;
}
