#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include "core/localization/lidar_loc/lidar_loc.h"
#include "core/localization/localization.h"
#include "core/localization/pose_graph/pgo.h"
#include "io/yaml_io.h"
#include "ui/pangolin_window.h"

namespace lightning::loc {

// ！ 构造函数
Localization::Localization(Options options) { options_ = options; }

// ！初始化函数
bool Localization::Init(const std::string& yaml_path, const std::string& global_map_path) {
    UL lock(global_mutex_);
    if (lidar_loc_ != nullptr) {
        // 若已经启动，则变为初始化
        Finish();
    }

    YAML_IO yaml(yaml_path);
    options_.with_ui_ = yaml.GetValue<bool>("system", "with_ui");

    /// lidar odom前端
    LaserMapping::Options opt_lio;
    // 非slam模式，只维持最近两帧，不会记录所有关键帧
    opt_lio.is_in_slam_mode_ = false;

    lio_ = std::make_shared<LaserMapping>(opt_lio);
    if (!lio_->Init(yaml_path)) {
        LOG(ERROR) << "failed to init lio";
        return false;
    }

    /// 激光定位
    LidarLoc::Options lidar_loc_options;
    lidar_loc_options.update_dynamic_cloud_ = yaml.GetValue<bool>("lidar_loc", "update_dynamic_cloud");
    lidar_loc_options.force_2d_ = yaml.GetValue<bool>("lidar_loc", "force_2d");
    lidar_loc_options.map_option_.enable_dynamic_polygon_ = false;
    lidar_loc_options.map_option_.map_path_ = global_map_path;
    lidar_loc_ = std::make_shared<LidarLoc>(lidar_loc_options);

    if (options_.with_ui_) {
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->SetCurrentScanSize(10);
        ui_->Init();
        lidar_loc_->SetUI(ui_);
    }

    lidar_loc_->Init(yaml_path);

    /// pose graph
    //QUES: 这里的PGO和建图是miao为啥不一样呢，难道这个不是以边和节点的形式？？类似与ceres的那种？
    pgo_ = std::make_shared<PGO>();
    pgo_->SetDebug(false);

    ///  各模块的异步调用
    options_.enable_lidar_loc_skip_ = yaml.GetValue<bool>("system", "enable_lidar_loc_skip");
    options_.enable_lidar_loc_rviz_ = yaml.GetValue<bool>("system", "enable_lidar_loc_rviz");
    options_.lidar_loc_skip_num_ = yaml.GetValue<int>("system", "lidar_loc_skip_num");
    options_.enable_lidar_odom_skip_ = yaml.GetValue<bool>("system", "enable_lidar_odom_skip");
    options_.lidar_odom_skip_num_ = yaml.GetValue<int>("system", "lidar_odom_skip_num");

    // NOTE: 这里仅保留一帧数据，以保证实时性，但是建图的时候不是
    lidar_odom_proc_cloud_.SetMaxSize(1);
    lidar_loc_proc_cloud_.SetMaxSize(1);

    lidar_odom_proc_cloud_.SetName("激光里程计");
    lidar_loc_proc_cloud_.SetName("激光定位");

    // 允许跳帧
    lidar_loc_proc_cloud_.SetSkipParam(options_.enable_lidar_loc_skip_, options_.lidar_loc_skip_num_);
    lidar_odom_proc_cloud_.SetSkipParam(options_.enable_lidar_odom_skip_, options_.lidar_odom_skip_num_);

    lidar_odom_proc_cloud_.SetProcFunc([this](CloudPtr cloud) { LidarOdomProcCloud(cloud); });
    lidar_loc_proc_cloud_.SetProcFunc([this](CloudPtr cloud) { LidarLocProcCloud(cloud); });

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.Start();
        lidar_loc_proc_cloud_.Start();
    }

    /// TODO: 发布
    /// QUES: 这里使用的后端优化和建图时的有什么区别吗？miao是添加节点和边，这边难道是ceres自己添加优化项的
    pgo_->SetHighFrequencyGlobalOutputHandleFunction([this](const LocalizationResult& res) {
        if (loc_result_.timestamp_ > 0) {
            double loc_fps = 1.0 / (res.timestamp_ - loc_result_.timestamp_);
            LOG_EVERY_N(INFO, 10) << "loc fps: " << loc_fps;
        }

        loc_result_ = res;

        if (tf_callback_ && loc_result_.valid_) {
            tf_callback_(loc_result_.ToGeoMsg());
        }

        if (ui_) {
            ui_->UpdateNavState(loc_result_.ToNavState());
            ui_->UpdateRecentPose(loc_result_.pose_);
        }
    });

    /// 预处理器
    preprocess_.reset(new PointCloudPreprocess());
    preprocess_->Blind() = yaml.GetValue<double>("fasterlio", "blind");
    preprocess_->TimeScale() = yaml.GetValue<double>("fasterlio", "time_scale");
    int lidar_type = yaml.GetValue<int>("fasterlio", "lidar_type");
    preprocess_->NumScans() = yaml.GetValue<int>("fasterlio", "scan_line");
    preprocess_->PointFilterNum() = yaml.GetValue<int>("fasterlio", "point_filter_num");

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
    }

    return true;
}

// 定位主流程 -- Lidar处理过程
void Localization::ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
    UL lock(global_mutex_);
    if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
        return;
    }

    // 串行模式
    // step1: 定位点云处理流程 -- 点云滤波和去除盲点
    CloudPtr laser_cloud(new PointCloudType);
    preprocess_->Process(cloud, laser_cloud);
    // QUES: 这边时间戳单位为ns，建图时我记得是ns
    laser_cloud->header.stamp = cloud->header.stamp.sec * 1e9 + cloud->header.stamp.nanosec;
    
    // step2: 定位点云处理流程 -- 异步处理线程添加数据（lio 和 lidar_loc_）
    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.AddMessage(laser_cloud);
    } else {
        LidarOdomProcCloud(laser_cloud);
    }
}

void Localization::ProcessLivoxLidarMsg(const livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
    UL lock(global_mutex_);
    if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
        return;
    }

    // 串行模式
    CloudPtr laser_cloud(new PointCloudType);
    preprocess_->Process(cloud, laser_cloud);
    laser_cloud->header.stamp = cloud->header.stamp.sec * 1e9 + cloud->header.stamp.nanosec;

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.AddMessage(laser_cloud);
    } else {
        LidarOdomProcCloud(laser_cloud);
    }
}

void Localization::LidarOdomProcCloud(CloudPtr cloud) {
    if (lio_ == nullptr) {
        return;
    }

    /// NOTE: 在NCLT这种数据集中，lio内部是有缓存的，它拿到的点云不一定是最新时刻的点云
    // QUES: 有个疑问，为什么做了两边雷达预处理过程？？？
    lio_->ProcessPointCloud2(cloud);
    if (!lio_->Run()) {
        return;
    }

    auto lo_state = lio_->GetState();

    /////////////////////////////////////////////////////////////////
    ////////////////////进入lidar_loc处理流程//////////////////////
    // note: 激光里程计输入
    lidar_loc_->ProcessLO(lo_state);
    pgo_->ProcessLidarOdom(lo_state);

    // LOG(INFO) << "LO pose: " << std::setprecision(12) << lo_state.timestamp_ << " "
    //           << lo_state.GetPose().translation().transpose();

    /// 获得lio的关键帧
    auto kf = lio_->GetKeyframe();

    if (kf == lio_kf_) {
        /// 关键帧未更新，那就只更新IMU状态

        // auto dr_state = lio_->GetState();
        // lidar_loc_->ProcessDR(dr_state);
        // pgo_->ProcessDR(dr_state);
        return;
    }

    lio_kf_ = kf;
    
    // scan是去畸变后的点云
    auto scan = lio_->GetScanUndist();

    if (options_.online_mode_) {
        lidar_loc_proc_cloud_.AddMessage(scan);  // NOTE: 这里是将去畸变后的点云添加到异步处理线程中，进行后续的定位处理
    } else {
        LidarLocProcCloud(scan);
    }
}

void Localization::LidarLocProcCloud(CloudPtr scan_undist) {
    // 展开lidar_loc的处理流程
    lidar_loc_->ProcessCloud(scan_undist);

    auto res = lidar_loc_->GetLocalizationResult();
    pgo_->ProcessLidarLoc(res);

    if (ui_) {
        // Twi with Til, here pose means Twl, thus Til=I
        ui_->UpdateScan(scan_undist, res.pose_);
    }

    if (loc_state_callback_) {
        auto loc_state = std::make_shared<std_msgs::msg::Int32>();
        loc_state->data = static_cast<int>(res.status_);
        LOG(INFO) << "loc_state: " << loc_state->data;
        loc_state_callback_(*loc_state);
    }
}

/// IMU处理流程
void Localization::ProcessIMUMsg(IMUPtr imu) {
    
    UL lock(global_mutex_);

    if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
        return;
    }

    double this_imu_time = imu->timestamp;
    if (last_imu_time_ > 0 && this_imu_time < last_imu_time_) {
        LOG(WARNING) << "IMU 时间异常：" << this_imu_time << ", last: " << last_imu_time_;
    }
    last_imu_time_ = this_imu_time;
    // step1: IMU处理流程--建图lio部分输入（传感器同步，其中一个eskf专项前端计算）
    /// 里程计处理IMU
    lio_->ProcessIMU(imu);

    /// 这里需要 IMU predict，否则没法process DR了
    auto dr_state = lio_->GetIMUState();

    // pose_is_ok_是指IMU状态是否已经初始化好了，如果没有初始化好，那么就不处理DR了
    if (!dr_state.pose_is_ok_) {
        return;
    }

    // /// 停车判定
    // constexpr auto kThVbrbStill = 0.05;  // 0.08;
    // constexpr auto kThOmegaStill = 0.05;

    // if (dr_state.GetVel().norm() < kThVbrbStill && imu->angular_velocity.norm() < kThOmegaStill) {
    //     dr_state.is_parking_ = true;
    //     dr_state.SetVel(Vec3d::Zero());
    // }

    /// 如果没有odm, 用lio替代DR

    // LOG(INFO) << "dr state: " << std::setprecision(12) << dr_state.timestamp_ << " "
    //           << dr_state.GetPose().translation().transpose()
    //           << ", q=" << dr_state.GetPose().unit_quaternion().coeffs().transpose();

    // step2: IMU处理流程--雷达定位dr数据输入和pose graph优化输入
    // imu DR航迹推算输入
    lidar_loc_->ProcessDR(dr_state);
    pgo_->ProcessDR(dr_state);
}

// void Localization::ProcessOdomMsg(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
//     UL lock(global_mutex_);
//
//     if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
//         return;
//     }
//     double this_odom_time = ToSec(odom_msg->header.stamp);
//     if (last_odom_time_ > 0 && this_odom_time < last_odom_time_) {
//         LOG(WARNING) << "Odom Time Abnormal:" << this_odom_time << ", last: " << last_odom_time_;
//     }
//     last_odom_time_ = this_odom_time;
//
//     lio_->ProcessOdometry(odom_msg);
//
//     if (!lio_->GetbOdomHF()) {
//         return;
//     }
//
//     auto dr_state = lio_->GetStateHF(mapping::FasterLioMapping::kHFStateOdomFiltered);
//
//     constexpr auto kThVbrbStill = 0.03;  // 0.08;
//     constexpr auto kThOmegaStill = 0.03;
//     if (dr_state.Getvwi().norm() < kThVbrbStill && dr_state.Getwii().norm() < kThOmegaStill) {
//         dr_state.is_parking_ = true;
//         dr_state.Setvwi(Vec3d::Zero());
//         dr_state.Setwii(Vec3d::Zero());
//     }
//
//     lidar_loc_->ProcessDR(dr_state);
//     pgo_->ProcessDR(dr_state);
// }

void Localization::Finish() {
    lidar_loc_->Finish();
    if (ui_) {
        ui_->Quit();
    }

    lidar_loc_proc_cloud_.Quit();
    lidar_odom_proc_cloud_.Quit();
}

void Localization::SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    UL lock(global_mutex_);
    /// 设置外部重定位的pose
    if (lidar_loc_) {
        lidar_loc_->SetInitialPose(SE3(q, t));
    }
}

void Localization::SetTFCallback(Localization::TFCallback&& callback) { tf_callback_ = callback; }

}  // namespace lightning::loc
