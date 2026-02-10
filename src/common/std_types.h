//
// Created by xiang on 25-3-12.
//

#ifndef LIGHTNING_STD_TYPES_H
#define LIGHTNING_STD_TYPES_H

#include <mutex>
#include <thread>

namespace lightning {
// 独占锁，同一时刻只允许单一访问，和shared_lock（读写锁）是对应的
using UL = std::unique_lock<std::mutex>;

}  // namespace lightning

#endif  // LIGHTNING_STD_TYPES_H
