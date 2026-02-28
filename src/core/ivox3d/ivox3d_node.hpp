#include <pcl/common/centroid.h>
#include <algorithm>
#include <cmath>
#include <list>
#include <vector>

#include "core/ivox3d/hilbert.hpp"
#include "core/lightning_math.hpp"

namespace lightning {

template <typename PointT, int dim = 3>
class IVoxNode {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct DistPoint;

    IVoxNode() = default;
    IVoxNode(const PointT& center, const float& side_length) {}  /// same with phc

    // 这个体素里面仅保留10个点
    void InsertPoint(const PointT& pt);

    inline bool Empty() const;

    inline std::size_t Size() const;

    inline PointT GetPoint(const std::size_t idx) const;

    int KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& point, const int& K,
                            const double& max_range);

   private:
    std::vector<PointT> points_;
};

template <typename PointT, int dim = 3>
class IVoxNodePhc {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct DistPoint;
    struct PhcCube;

    IVoxNodePhc() = default;
    IVoxNodePhc(const PointT& center, const float& side_length, const int& phc_order = 6);

    void InsertPoint(const PointT& pt);

    void ErasePoint(const PointT& pt, const double erase_distance_th_);

    inline bool Empty() const;

    inline std::size_t Size() const;

    PointT GetPoint(const std::size_t idx) const;

    bool NNPoint(const PointT& cur_pt, DistPoint& dist_point) const;

    int KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& cur_pt, const int& K = 5,
                            const double& max_range = 5.0);

   private:
    uint32_t CalculatePhcIndex(const PointT& pt) const;

   private:
    std::vector<PhcCube> phc_cubes_;

    PointT center_;
    float side_length_ = 0;
    int phc_order_ = 6;
    float phc_side_length_ = 0;
    float phc_side_length_inv_ = 0;
    Eigen::Matrix<float, dim, 1> min_cube_;
};

template <typename PointT, int dim>
struct IVoxNode<PointT, dim>::DistPoint {
    double dist = 0;
    IVoxNode* node = nullptr;
    int idx = 0;

    DistPoint() = default;
    DistPoint(const double d, IVoxNode* n, const int i) : dist(d), node(n), idx(i) {}

    PointT Get() { return node->GetPoint(idx); }

    inline bool operator()(const DistPoint& p1, const DistPoint& p2) { return p1.dist < p2.dist; }

    inline bool operator<(const DistPoint& rhs) { return dist < rhs.dist; }
};

// 不使用PHC的ivox_node时，体素内就保留10个点，多了直接删除
template <typename PointT, int dim>
void IVoxNode<PointT, dim>::InsertPoint(const PointT& pt) {
    points_.template emplace_back(pt);
    if (points_.size() >= 10) {
        points_.erase(points_.begin());
    }
}

template <typename PointT, int dim>
bool IVoxNode<PointT, dim>::Empty() const {
    return points_.empty();
}

template <typename PointT, int dim>
std::size_t IVoxNode<PointT, dim>::Size() const {
    return points_.size();
}

template <typename PointT, int dim>
PointT IVoxNode<PointT, dim>::GetPoint(const std::size_t idx) const {
    return points_[idx];
}

// 仅保留k个，也是要队列中顺序的一段
template <typename PointT, int dim>
int IVoxNode<PointT, dim>::KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& point, const int& K,
                                               const double& max_range) {
    std::size_t old_size = dis_points.size();
// #define INNER_TIMER   // 内部定时器，不确定是啥意思
#ifdef INNER_TIMER
    static std::unordered_map<std::string, std::vector<int64_t>> stats;
    if (stats.empty()) {
        stats["dis"] = std::vector<int64_t>();
        stats["put"] = std::vector<int64_t>();
        stats["nth"] = std::vector<int64_t>();
    }
#endif

    for (const auto& pt : points_) {
#ifdef INNER_TIMER
        auto t0 = std::chrono::high_resolution_clock::now();
#endif
        double d = math::distance2(pt, point);
#ifdef INNER_TIMER
        auto t1 = std::chrono::high_resolution_clock::now();
#endif
        // std::vector::data() 是返回vector第一个元素的指针
        /**
         * std::vector<int> test(5);
         * int *a = test.data();
         * a++;
         * a[2] = 10;
         *
         * test: 0 0 0 10 0
         */
        if (d < max_range * max_range) {
            dis_points.template emplace_back(DistPoint(d, this, &pt - points_.data()));
        }
#ifdef INNER_TIMER
        auto t2 = std::chrono::high_resolution_clock::now();

        auto dis = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        stats["dis"].emplace_back(dis);
        auto put = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        stats["put"].emplace_back(put);
#endif
    }

#ifdef INNER_TIMER
    auto t1 = std::chrono::high_resolution_clock::now();
#endif
    // sort by distance
    // 仅保留K个距离该点最小的点
    if (old_size + K >= dis_points.size()) {
        // 点数过少时不做处理

        // 下面当点数过多时，做排序和resize操作
    } else {
        // 函数原型
        /**
         * template<class RandomAccessIterator>
         * void nth_element(RandomAccessIterator first, RandomAccessIterator nth, RandomAccessIterator last);
         * fisrt 指向范围的第一个元素的随机访问迭代器
         * nth 指向目标位置的随机访问迭代器
         * last 指向范围最后一个元素之后的迭代器
         * 作用：将nth位置处的元素放到整个队列中，要求之前的内容全比该元素小，之后的内容全比该元素大
         */
        std::nth_element(dis_points.begin() + old_size, dis_points.begin() + old_size + K - 1, dis_points.end());
        dis_points.resize(old_size + K);
    }

#ifdef INNER_TIMER
    auto t2 = std::chrono::high_resolution_clock::now();
    auto nth = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    stats["nth"].emplace_back(nth);

    constexpr int STAT_PERIOD = 100000;
    if (!stats["nth"].empty() && stats["nth"].size() % STAT_PERIOD == 0) {
        for (auto& it : stats) {
            const std::string& key = it.first;
            std::vector<int64_t>& stat = it.second;
            int64_t sum_ = std::accumulate(stat.begin(), stat.end(), 0);
            int64_t num_ = stat.size();
            stat.clear();
            std::cout << "inner_" << key << "(ns): sum=" << sum_ << " num=" << num_ << " ave=" << 1.0 * sum_ / num_
                      << " ave*n=" << 1.0 * sum_ / STAT_PERIOD << std::endl;
        }
    }
#endif

    return dis_points.size();
}

template <typename PointT, int dim>
struct IVoxNodePhc<PointT, dim>::DistPoint {
    double dist = 0;
    IVoxNodePhc* node = nullptr;
    int idx = 0;

    DistPoint() {}
    DistPoint(const double d, IVoxNodePhc* n, const int i) : dist(d), node(n), idx(i) {}

    PointT Get() { return node->GetPoint(idx); }

    inline bool operator()(const DistPoint& p1, const DistPoint& p2) { return p1.dist < p2.dist; }

    inline bool operator<(const DistPoint& rhs) { return dist < rhs.dist; }
};

template <typename PointT, int dim>
struct IVoxNodePhc<PointT, dim>::PhcCube {
    uint32_t idx = 0;
    // learn: pcl::CentroidPoint 增量式计算点云质心 add增加点云 get计算点云质心

    pcl::CentroidPoint<PointT> mean;

    PhcCube(uint32_t index, const PointT& pt) { mean.add(pt); }

    void AddPoint(const PointT& pt) { mean.add(pt); }

    PointT GetPoint() const {
        PointT pt;
        mean.get(pt);
        return std::move(pt);
    }
};

/**
 * @brief
 *
 * @param center    第一个激光点坐标
 * @param side_length  体素大小（分辨率）
 * @param phc_order 伪希尔伯特曲线中的参数，用2^K次方来填满这个划分的体素，默认为6
 * @return template <typename PointT, int dim>
 */
template <typename PointT, int dim>
IVoxNodePhc<PointT, dim>::IVoxNodePhc(const PointT& center, const float& side_length, const int& phc_order)
    : center_(center), side_length_(side_length), phc_order_(phc_order) {
    assert(phc_order <= 8);
    phc_side_length_ = side_length_ / (std::pow(2, phc_order_));  // 体素内每个单元的分辨率
    phc_side_length_inv_ = (std::pow(2, phc_order_)) / side_length_;
    min_cube_ = center_.getArray3fMap() - side_length / 2.0;  // 体素内左下角坐标
    phc_cubes_.reserve(64);
}

template <typename PointT, int dim>
void IVoxNodePhc<PointT, dim>::InsertPoint(const PointT& pt) {
    uint32_t idx = CalculatePhcIndex(pt);

    PhcCube cube{idx, pt};
    // std::lower_bound在有序区间中查找第一个>=给定值的位置
    auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                               [](const PhcCube& a, const PhcCube& b) { return a.idx < b.idx; });

    if (it == phc_cubes_.end()) {
        // 没有比这个id大的就放到容器最后面
        phc_cubes_.emplace_back(cube);
    } else {
        if (it->idx == idx) {
            // 如果id相同，就直接求质心
            it->AddPoint(pt);
        } else {
            // 如果找到了这个id，就把新id插进来
            phc_cubes_.insert(it, cube);
        }
    }
}

// 直接删除点
template <typename PointT, int dim>
void IVoxNodePhc<PointT, dim>::ErasePoint(const PointT& pt, const double erase_distance_th_) {
    uint32_t idx = CalculatePhcIndex(pt);

    PhcCube cube{idx, pt};
    auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                               [](const PhcCube& a, const PhcCube& b) { return a.idx < b.idx; });

    if (erase_distance_th_ > 0) {
    }
    if (it != phc_cubes_.end() && it->idx == idx) {
        phc_cubes_.erase(it);
    }
}

template <typename PointT, int dim>
bool IVoxNodePhc<PointT, dim>::Empty() const {
    return phc_cubes_.empty();
}

template <typename PointT, int dim>
std::size_t IVoxNodePhc<PointT, dim>::Size() const {
    return phc_cubes_.size();
}

template <typename PointT, int dim>
PointT IVoxNodePhc<PointT, dim>::GetPoint(const std::size_t idx) const {
    return phc_cubes_[idx].GetPoint();
}

// 先根据索引找出大于等于索引的点
/**
 * 情况a：如果没找到大于等于索引的点，就用最后一个点
 * 情况b：如果发现第一个点满足，就用第一个点
 * 情况c：在内部能找到，就看前后那个点的距离小就用哪个
 */
template <typename PointT, int dim>
bool IVoxNodePhc<PointT, dim>::NNPoint(const PointT& cur_pt, DistPoint& dist_point) const {
    if (phc_cubes_.empty()) {
        return false;
    }
    uint32_t cur_idx = CalculatePhcIndex(cur_pt);
    PhcCube cube{cur_idx, cur_pt};
    auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                               [](const PhcCube& a, const PhcCube& b) { return a.idx < b.idx; });

    if (it == phc_cubes_.end()) {
        it--;
        dist_point = DistPoint(distance2(cur_pt, it->GetPoint()), this, it - phc_cubes_.begin());
    } else if (it == phc_cubes_.begin()) {
        dist_point = DistPoint(distance2(cur_pt, it->GetPoint()), this, it - phc_cubes_.begin());
    } else {
        auto last_it = it;
        last_it--;
        double d1 = distance2(cur_pt, it->GetPoint());
        double d2 = distance2(cur_pt, last_it->GetPoint());
        if (d1 > d2) {
            dist_point = DistPoint(d2, this, it - phc_cubes_.begin());
        } else {
            dist_point = DistPoint(d1, this, it - phc_cubes_.begin());
        }
    }

    return true;
}

// 只保留K个
template <typename PointT, int dim>
int IVoxNodePhc<PointT, dim>::KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& cur_pt,
                                                  const int& K, const double& max_range) {
    uint32_t cur_idx = CalculatePhcIndex(cur_pt);
    PhcCube cube{cur_idx, cur_pt};
    auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                               [](const PhcCube& a, const PhcCube& b) { return a.idx < b.idx; });

    const int max_search_cube_side_length = std::pow(2, std::ceil(std::log2(max_range * phc_side_length_inv_)));
    const int max_search_idx_th =
        8 * max_search_cube_side_length * max_search_cube_side_length * max_search_cube_side_length;

    // 计算体素内单元与当前激光点的距离
    auto create_dist_point = [&cur_pt, this](typename std::vector<PhcCube>::const_iterator forward_it) {
        double d = distance2(forward_it->GetPoint(), cur_pt);
        return DistPoint(d, this, forward_it - phc_cubes_.begin());
    };

    typename std::vector<PhcCube>::const_iterator forward_it(it);
    typename std::vector<PhcCube>::const_reverse_iterator backward_it(it);
    if (it != phc_cubes_.end()) {
        dis_points.emplace_back(create_dist_point(it));
        forward_it++;
    }
    if (backward_it != phc_cubes_.rend()) {
        backward_it++;
    }

    auto forward_reach_boundary = [&]() {
        return forward_it == phc_cubes_.end() || forward_it->idx - cur_idx > max_search_idx_th;
    };
    auto backward_reach_boundary = [&]() {
        return backward_it == phc_cubes_.rend() || cur_idx - backward_it->idx > max_search_idx_th;
    };
    // 优先往前搜索
    while (!forward_reach_boundary() && !backward_reach_boundary()) {
        if (forward_it->idx - cur_idx > cur_idx - backward_it->idx) {
            dis_points.emplace_back(create_dist_point(forward_it));
            forward_it++;
        } else {
            // 返回该反向迭代器对应的"正向迭代器",但指向的是“下一个元素”
            // QUES: 那这里是不是有点问题
            dis_points.emplace_back(create_dist_point(backward_it.base()));
            backward_it++;
        }
        if (dis_points.size() > K) {
            break;
        }
    }
    // 如果往前搜索完成了，就往后搜索
    if (forward_reach_boundary()) {
        while (!backward_reach_boundary() && dis_points.size() < K) {
            dis_points.emplace_back(create_dist_point(backward_it.base()));
            backward_it++;
        }
    }
    // 如果往后搜索完成了，就往前搜索
    if (backward_reach_boundary()) {
        while (!forward_reach_boundary() && dis_points.size() < K) {
            dis_points.emplace_back(create_dist_point(forward_it));
            forward_it++;
        }
    }

    return dis_points.size();
}

template <typename PointT, int dim>
uint32_t IVoxNodePhc<PointT, dim>::CalculatePhcIndex(const PointT& pt) const {
    Eigen::Matrix<float, dim, 1> eposf =
        (pt.getVector3fMap() - min_cube_) * phc_side_length_inv_;  // 计算一个体素内一个单元格的索引 float
    Eigen::Matrix<int, dim, 1> eposi = eposf.template cast<int>();  // 转换为int类型
    for (int i = 0; i < dim; ++i) {
        if (eposi(i, 0) < 0) {
            eposi(i, 0) = 0;
        }
        if (eposi(i, 0) > std::pow(2, phc_order_)) {
            eposi(i, 0) = std::pow(2, phc_order_) - 1;
        }
    }
    std::array<uint8_t, 3> apos{eposi(0), eposi(1), eposi(2)};
    // 下面所有内容转换为伪希尔伯特计算连接索引的东西，具体不用展开
    std::array<uint8_t, 3> tmp = hilbert::v2::PositionToIndex(apos);

    uint32_t idx = (uint32_t(tmp[0]) << 16) + (uint32_t(tmp[1]) << 8) + (uint32_t(tmp[2]));
    return idx;
}

}  // namespace lightning
