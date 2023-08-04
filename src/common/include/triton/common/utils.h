#pragma once

#include <algorithm>
#include <chrono>
#include <ctime>
#include <random>

#include <dirent.h>
#include <pthread.h>
#include <sys/types.h>
#include <cstdlib>
#include <memory>
#include <string>

#include <boost/thread/mutex.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace dataelem { namespace common {

namespace Random {

namespace {
thread_local std::mt19937_64 generator(std::random_device{}());
}

// Random number with normal distribution and mean of zero
inline double
random_normal(double standard_deviation)
{
  using nd = std::normal_distribution<double>;
  thread_local static auto dist = nd{};
  return dist(generator, nd::param_type{0.0, standard_deviation});
}

// Random number with inclusive range
inline double
random_real(double min, double max)
{
  using urd = std::uniform_real_distribution<double>;
  thread_local static auto dist = urd{};
  return dist(generator, urd::param_type{min, max});
}

inline int
random_integer(int min, int max)
{
  using uid = std::uniform_int_distribution<int>;
  thread_local static auto dist = uid{};
  return dist(generator, uid::param_type{min, max});
}

// Return true with given probability
inline bool
success_probability(double probability)
{
  return random_real(0.0, 1.0) < probability;
}

// Return true with probability 50%
inline bool
coin_flip()
{
  return success_probability(0.5);
}

// Shuffles the order of the list
template <class List>
void
shuffle(List& list)
{
  std::shuffle(list.begin(), list.end(), generator);
}

class UUIDS {
 public:
  std::string get()
  {
    boost::uuids::uuid uuid;
    {
      boost::mutex::scoped_lock lock(_mutex);
      uuid = _gen();
    }
    return boost::uuids::to_string(uuid);
  }

 private:
  boost::mutex _mutex;
  boost::uuids::random_generator _gen;
};

};  // namespace Random

class Timer {
 public:
  Timer() { _tstart = std::chrono::system_clock::now(); }
  ~Timer() {}

  void tic() { _tstart = std::chrono::system_clock::now(); }

  int toc()
  {
    auto tstop = std::chrono::system_clock::now();
    int elapse =
        std::chrono::duration_cast<std::chrono::milliseconds>(tstop - _tstart)
            .count();
    tic();
    return elapse;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> _tstart;
};

template <typename Clock>
class CpuTimer {
 public:
  using clock_type = Clock;

  void tic() { mStart = Clock::now(); }
  float toc()
  {
    mStop = Clock::now();
    float mMs =
        std::chrono::duration<float, std::milli>{mStop - mStart}.count();
    tic();
    return mMs;
  }

 private:
  std::chrono::time_point<Clock> mStart, mStop;
};  // class CpuTimer

using PreciseCpuTimer = CpuTimer<std::chrono::high_resolution_clock>;

// thread safe Singleton
template <typename T>
class Singleton {
 private:
  static T* _instance_ptr;
  static pthread_once_t _once;

 private:
  Singleton();
  ~Singleton();

 public:
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;

  static T* getInstance()
  {
    pthread_once(&_once, init);
    return _instance_ptr;
  }

  static void init() { _instance_ptr = new T(); }

  static void release()
  {
    if (_instance_ptr != nullptr) {
      delete _instance_ptr;
      _instance_ptr = nullptr;
    }
  }
};

template <typename T>
T* Singleton<T>::_instance_ptr = nullptr;

template <typename T>
pthread_once_t Singleton<T>::_once = PTHREAD_ONCE_INIT;

}}  // namespace dataelem::common
